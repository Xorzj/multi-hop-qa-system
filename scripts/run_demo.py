"""端到端 Demo：从 .docx 文档到多跳推理问答的完整流程。

用法:
    uv run python -m scripts.run_demo                          # 使用默认文档和问题
    uv run python -m scripts.run_demo --doc path/to/file.docx  # 指定文档
    uv run python -m scripts.run_demo --question "你的问题"     # 指定问题
    uv run python -m scripts.run_demo --skip-offline  # 跳过离线阶段

流程:
    ┌─────────────────────── 离线阶段 ───────────────────────┐
    │ .docx → 语义切分 → 实体抽取 → 三元组抽取 → 知识图谱构建 │
    └────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────── 在线阶段 ───────────────────────┐
    │ 问题 → 意图解析 → 图谱检索 → 多跳推理 → 上下文组装 → 答案 │
    └────────────────────────────────────────────────────────┘
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

from src.common.config import Config, load_config
from src.common.logger import get_logger
from src.data_processing.document_loader import DocumentLoader
from src.data_processing.entity_extractor import EntityExtractor
from src.data_processing.triple_extractor import TripleExtractor
from src.knowledge_graph.graph_builder import GraphBuilder
from src.knowledge_graph.graph_retriever import GraphRetriever
from src.knowledge_graph.neo4j_client import Neo4jClient
from src.llm.base_client import BaseLLMClient
from src.llm.client_factory import create_llm_client
from src.qa_engine.answer_generator import AnswerGenerator
from src.qa_engine.context_assembler import ContextAssembler
from src.qa_engine.question_parser import QuestionParser
from src.reasoning.reasoning_orchestrator import ReasoningOrchestrator

logger = get_logger(__name__)

# ──────────────────── 默认配置 ────────────────────

DEFAULT_DOC = "test_documents/03.docx"
DEFAULT_QUESTIONS = [
    "DWDM和WDM是什么关系？",
    "SDH网络中使用了哪些复用技术？",
    "光纤通信系统包含哪些关键组件？",
]

# 离线阶段最大处理段落数（用于 demo 快速演示，设 None 则处理全文）
MAX_SECTIONS_FOR_DEMO: int | None = 10


# ──────────────────── 离线阶段 ────────────────────


async def run_offline_pipeline(
    doc_path: str,
    config: Config,
    neo4j_client: Neo4jClient,
    llm_client: BaseLLMClient,
) -> dict[str, int]:
    """离线流水线：从 .docx 构建知识图谱。

    Returns:
        包含 nodes/relations 计数的统计字典。
    """
    print("\n" + "=" * 60)
    print("📄 离线阶段：文档 → 知识图谱")
    print("=" * 60)

    t0 = time.time()

    # ① 文档加载
    print(f"\n[1/5] 加载文档: {doc_path}")
    loader = DocumentLoader()
    doc = loader.load(doc_path)
    print(f"      文档长度: {len(doc.content):,} 字符")

    # ② 语义切分
    print("\n[2/5] 语义切分...")
    extraction_cfg = config.extraction
    sections = loader.split_into_sections(
        doc.content,
        max_chunk_size=extraction_cfg.entity_chunk_size,
    )
    total_sections = len(sections)
    if MAX_SECTIONS_FOR_DEMO is not None and total_sections > MAX_SECTIONS_FOR_DEMO:
        limit = MAX_SECTIONS_FOR_DEMO
        print(f"      共 {total_sections} 段落，Demo 模式只处理前 {limit} 段")
        sections = sections[:MAX_SECTIONS_FOR_DEMO]
    else:
        print(f"      共 {total_sections} 段落")

    for i, sec in enumerate(sections[:5]):
        preview = sec.content[:60].replace("\n", " ")
        heading = " > ".join(sec.heading_chain) if sec.heading_chain else "无标题"
        print(f"      段落 {i + 1}: [{heading}] {preview}...")
    if len(sections) > 5:
        print(f"      ... 还有 {len(sections) - 5} 段")
    # ③ 实体抽取
    print("\n[3/5] 实体抽取（使用 LLM）...")
    entity_extractor = EntityExtractor(
        llm_client,
        extraction_config=extraction_cfg,
    )
    entities = await entity_extractor.extract(sections)
    print(f"      抽取到 {len(entities)} 个实体:")
    for ent in entities[:10]:
        print(f"        - {ent.name} ({ent.entity_type})")
    if len(entities) > 10:
        print(f"        ... 还有 {len(entities) - 10} 个实体")

    # ④ 三元组抽取
    print("\n[4/5] 三元组抽取（使用 LLM）...")
    triple_extractor = TripleExtractor(
        llm_client,
        extraction_config=extraction_cfg,
    )
    triples = await triple_extractor.extract(sections, entities)
    print(f"      抽取到 {len(triples)} 个三元组:")
    for tri in triples[:10]:
        print(f"        - {tri.subject} --[{tri.predicate}]--> {tri.object}")
    if len(triples) > 10:
        print(f"        ... 还有 {len(triples) - 10} 个三元组")

    # ⑤ 知识图谱构建
    print("\n[5/5] 构建知识图谱（写入 Neo4j）...")
    # 先清空旧数据
    await neo4j_client.execute("MATCH (n) DETACH DELETE n")
    print("      已清空旧图谱数据")

    builder = GraphBuilder(neo4j_client, auto_create_missing_nodes=True)
    stats = await builder.build_from_extraction(entities, triples)
    print(f"      节点: {stats.nodes_created} 个")
    print(f"      关系: {stats.relations_created} 个")

    # 验证
    node_result = await neo4j_client.execute("MATCH (n) RETURN count(n) as cnt")
    rel_result = await neo4j_client.execute("MATCH ()-[r]->() RETURN count(r) as cnt")
    node_count = node_result[0]["cnt"] if node_result else 0
    rel_count = rel_result[0]["cnt"] if rel_result else 0

    elapsed = time.time() - t0
    print(f"\n✅ 离线阶段完成！耗时 {elapsed:.1f} 秒")
    print(f"   图谱规模: {node_count} 节点, {rel_count} 关系")

    return {"nodes": node_count, "relations": rel_count, "elapsed_s": int(elapsed)}


# ──────────────────── 在线阶段 ────────────────────


async def run_online_qa(
    question: str,
    neo4j_client: Neo4jClient,
    llm_client: BaseLLMClient,
) -> None:
    """在线问答：问题 → 多跳推理 → 答案。"""
    print(f"\n{'─' * 60}")
    print(f"❓ 问题: {question}")
    print("─" * 60)

    t0 = time.time()

    # ① 问题解析
    print("\n[1/4] 解析问题意图...")
    parser = QuestionParser(llm_client)
    parsed = await parser.parse(question)
    print(f"      意图: {parsed.intent.value}")
    print(f"      识别实体: {parsed.entities}")
    if parsed.relation_hints:
        print(f"      关系提示: {parsed.relation_hints}")

    # ② 多跳推理
    print("\n[2/4] 多跳推理（图谱检索）...")
    retriever = GraphRetriever(neo4j_client)
    orchestrator = ReasoningOrchestrator(retriever, llm_client)
    evidence = await orchestrator.reason(parsed)
    print(f"      推理路径: {evidence.get_path_description()}")
    print(f"      证据节点: {len(evidence.nodes)} 个")
    print(f"      证据边: {len(evidence.edges)} 个")
    print(f"      推理步数: {len(evidence.steps)} 步")
    print(f"      置信度: {evidence.total_confidence:.2%}")

    # ③ 上下文组装
    print("\n[3/4] 组装推理上下文...")
    assembler = ContextAssembler()
    context = assembler.assemble(question, evidence, include_reasoning=True)
    print(f"      上下文长度: {len(context.prompt)} 字符")
    print(f"      推理步骤数: {len(context.reasoning_steps)}")

    # ④ 答案生成
    print("\n[4/4] 生成答案（使用 LLM）...")
    generator = AnswerGenerator(llm_client)
    answer = await generator.generate(context, include_reasoning=True)

    elapsed = time.time() - t0

    # 输出结果
    print(f"\n{'═' * 60}")
    print("💡 回答:")
    print(f"{'═' * 60}")
    print(answer.answer)
    print("\n📊 元信息:")
    print(f"   置信度: {answer.confidence:.2%}")
    print(f"   耗时: {elapsed:.1f} 秒 ({answer.latency_ms:.0f}ms 生成)")
    if answer.reasoning_steps:
        print("   推理步骤:")
        for step in answer.reasoning_steps:
            print(f"     • {step}")


# ──────────────────── 主流程 ────────────────────


async def main() -> None:
    """端到端 Demo 主函数。"""
    arg_parser = argparse.ArgumentParser(
        description="端到端 Demo：从 .docx 文档到多跳推理问答",
    )
    arg_parser.add_argument(
        "--doc",
        default=DEFAULT_DOC,
        help=f"输入 .docx 文档路径 (默认: {DEFAULT_DOC})",
    )
    arg_parser.add_argument(
        "--question",
        default=None,
        help="指定问题（不指定则使用预设问题列表）",
    )
    arg_parser.add_argument(
        "--skip-offline",
        action="store_true",
        help="跳过离线阶段（图谱已构建时使用）",
    )
    arg_parser.add_argument(
        "--max-sections",
        type=int,
        default=10,
        help="最大处理段落数 (默认: 10, 0=全部)",
    )
    args = arg_parser.parse_args()

    # 更新段落限制
    global MAX_SECTIONS_FOR_DEMO  # noqa: PLW0603
    MAX_SECTIONS_FOR_DEMO = args.max_sections if args.max_sections != 0 else None

    # 验证文档存在
    if not args.skip_offline:
        doc_file = Path(args.doc)
        if not doc_file.exists():
            print(f"❌ 文档不存在: {args.doc}")
            sys.exit(1)

    print("=" * 60)
    print("🚀 多跳推理知识图谱问答系统 — 端到端 Demo")
    print("=" * 60)

    # 加载配置
    config = load_config()
    print(f"模型: {config.llm.model_path}")
    print(f"图数据库: {config.graph.uri}")

    # 初始化组件
    neo4j_client = Neo4jClient(config.graph)
    llm_client = create_llm_client(config.llm)

    try:
        # 连接 Neo4j
        print("\n🔗 连接 Neo4j...")
        await neo4j_client.connect()
        print("   ✅ Neo4j 已连接")

        # 启动 LLM
        print("\n🤖 加载 LLM 模型...")
        await llm_client.start()  # type: ignore[attr-defined]
        print(f"   ✅ 模型已加载 ({config.llm.model_path})")

        # ═══════════════ 离线阶段 ═══════════════
        if not args.skip_offline:
            graph_stats = await run_offline_pipeline(
                args.doc,
                config,
                neo4j_client,
                llm_client,
            )
            if graph_stats["nodes"] == 0:
                print("\n⚠️  图谱为空，无法进行问答。请检查抽取结果。")
                return
        else:
            # 检查图谱是否已有数据
            node_result = await neo4j_client.execute("MATCH (n) RETURN count(n) as cnt")
            node_count = node_result[0]["cnt"] if node_result else 0
            if node_count == 0:
                print("\n⚠️  图谱为空！请先运行离线阶段（去掉 --skip-offline）。")
                return
            print(f"\n⏭️  跳过离线阶段（图谱已有 {node_count} 个节点）")

        # ═══════════════ 在线阶段 ═══════════════
        print("\n" + "=" * 60)
        print("🧠 在线阶段：多跳推理问答")
        print("=" * 60)

        questions = [args.question] if args.question else DEFAULT_QUESTIONS

        for i, question in enumerate(questions):
            if len(questions) > 1:
                print(f"\n{'━' * 60}")
                print(f"  问题 {i + 1}/{len(questions)}")
                print("━" * 60)
            await run_online_qa(question, neo4j_client, llm_client)

        # ═══════════════ 交互模式 ═══════════════
        if not args.question:
            print("\n" + "=" * 60)
            print("💬 进入交互模式（输入问题，输入 'quit' 退出）")
            print("=" * 60)

            while True:
                try:
                    user_input = input("\n❓ 你的问题: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n\n👋 再见！")
                    break

                if not user_input:
                    continue
                if user_input.lower() in ("quit", "exit", "q", "退出"):
                    print("\n👋 再见！")
                    break

                await run_online_qa(user_input, neo4j_client, llm_client)

    finally:
        # 清理资源
        print("\n🧹 清理资源...")
        await llm_client.stop()  # type: ignore[attr-defined]
        await neo4j_client.close()
        print("   ✅ 资源已释放")


if __name__ == "__main__":
    asyncio.run(main())
