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
import logging
import sys
import time
from datetime import datetime
from io import TextIOBase
from pathlib import Path

from src.common.config import Config, load_config
from src.common.logger import get_logger, setup_logging
from src.data_processing.document_loader import DocumentLoader
from src.data_processing.entity_extractor import EntityExtractor
from src.data_processing.schema_inducer import DomainSchema
from src.data_processing.triple_extractor import TripleExtractor
from src.knowledge_graph.graph_builder import GraphBuilder
from src.knowledge_graph.graph_retriever import GraphRetriever
from src.knowledge_graph.neo4j_client import Neo4jClient
from src.llm.base_client import BaseLLMClient
from src.llm.client_factory import create_llm_client
from src.qa_engine.answer_generator import AnswerGenerator
from src.qa_engine.context_assembler import ContextAssembler
from src.qa_engine.query_rewriter import QueryRewriter
from src.qa_engine.question_parser import QuestionParser
from src.reasoning.evidence_chain import EvidenceChain
from src.reasoning.reasoning_orchestrator import ReasoningOrchestrator

logger = get_logger(__name__)


class _TeeStream(TextIOBase):
    """Write to both a stream (stdout) and a log file simultaneously."""

    def __init__(self, stream: TextIOBase, log_file: TextIOBase) -> None:
        self.stream = stream
        self.log_file = log_file

    def write(self, data: str) -> int:
        self.stream.write(data)
        self.log_file.write(data)
        return len(data)

    def flush(self) -> None:
        self.stream.flush()
        self.log_file.flush()

    def fileno(self) -> int:
        return self.stream.fileno()

    @property
    def encoding(self) -> str:
        return getattr(self.stream, "encoding", None) or "utf-8"

    def readable(self) -> bool:
        return False

    def writable(self) -> bool:
        return True


# ──────────────────── 默认配置 ────────────────────

DEFAULT_DOC = "test_documents/demo.docx"
DEFAULT_QUESTIONS = [
    "DWDM和WDM是什么关系？",
    "SDH网络中使用了哪些复用技术？",
    "光纤通信系统包含哪些关键组件？",
]

# 离线阶段最大处理段落数（用于 demo 快速演示，设 None 则处理全文）
MAX_SECTIONS_FOR_DEMO: int | None = None


# ──────────────────── 格式化工具 ────────────────────

_W = 60  # 输出框宽度


def _hr(char: str = "─", width: int = _W) -> str:
    """Return a horizontal rule string."""
    return char * width


def _phase(step: str) -> float:
    """Print phase header and return start time."""
    print(f"\n  ⏳ {step}")
    return time.time()


def _done(step: str, started: float) -> None:
    """Print phase completion with elapsed time."""
    elapsed = time.time() - started
    print(f"  ✅ {step} ({elapsed:.1f}s)")


def _box(title: str, lines: list[str], indent: int = 5) -> None:
    """Print content in a bordered box with │ left border."""
    pad = " " * indent
    bar_w = max(_W - len(title) - 4, 10)
    print(f"{pad}┌─ {title} {_hr(width=bar_w)}")
    for line in lines:
        print(f"{pad}│ {line}")
    print(f"{pad}└{_hr(width=_W)}")


def _truncate(text: str, max_len: int = 50) -> str:
    """Truncate text with ellipsis for display."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _compact_rows(
    items: list[str], *, max_items: int = 0, row_width: int = 52
) -> list[str]:
    """Pack items into rows joined by ' · ', wrapping at row_width chars."""
    display = items[:max_items] if max_items else items
    rows: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for item in display:
        needed = len(item) + (3 if buf else 0)  # ' · ' separator
        if buf and buf_len + needed > row_width:
            rows.append(" · ".join(buf))
            buf = []
            buf_len = 0
        buf.append(item)
        buf_len += needed
    if buf:
        rows.append(" · ".join(buf))
    if max_items and len(items) > max_items:
        rows.append(f"… 共 {len(items)} 个")
    return rows


def _format_evidence(evidence: EvidenceChain, *, verbose: bool = False) -> list[str]:
    """Format evidence chain as human-readable lines (not XML)."""
    if not evidence.edges:
        return ["(无推理边)"]

    lines: list[str] = []

    # Group edges by hop
    edges_by_hop: dict[int, list] = {}
    for edge in evidence.edges:
        edges_by_hop.setdefault(edge.hop, []).append(edge)

    for hop_num in sorted(edges_by_hop):
        for edge in edges_by_hop[hop_num]:
            conf = f"  ({edge.confidence:.0%})" if verbose else ""
            lines.append(
                f"hop {hop_num}  {edge.source} ──[{edge.relation_type}]──→ {edge.target}{conf}"
            )

    # Verbose: show reasoning steps
    if verbose and evidence.steps:
        lines.append(_hr("┄", 48))
        lines.append("推理步骤:")
        for step in evidence.steps:
            nodes = "、".join(step.nodes_explored) if step.nodes_explored else "?"
            rel = step.relation_used or "?"
            text = step.reasoning or f"通过「{rel}」探索 {nodes}"
            lines.append(f"  {step.hop_number}. {text}")

    # Summary
    lines.append(_hr("┄", 48))
    lines.append(
        f"节点: {len(evidence.nodes)}  边: {len(evidence.edges)}"
        f"  步数: {len(evidence.steps)}  置信度: {evidence.total_confidence:.1%}"
    )
    return lines


# ──────────────────── 离线阶段 ────────────────────


async def run_offline_pipeline(
    doc_path: str,
    config: Config,
    neo4j_client: Neo4jClient,
    llm_client: BaseLLMClient,
    *,
    verbose: bool = False,
) -> dict[str, int]:
    """离线流水线：从 .docx 构建知识图谱。"""
    print(f"\n{'═' * _W}")
    print("  📄 离线阶段：文档 → 知识图谱")
    print(f"{'═' * _W}")

    t0 = time.time()

    # ① 文档加载
    phase = _phase(f"[1/5] 加载文档: {Path(doc_path).name}")
    loader = DocumentLoader()
    doc = loader.load(doc_path)
    print(f"     → {len(doc.content):,} 字符")
    _done("[1/5] 文档加载", phase)

    # ② 语义切分
    phase = _phase("[2/5] 语义切分")
    extraction_cfg = config.extraction
    sections = loader.split_into_sections(
        doc.content,
        max_chunk_size=extraction_cfg.entity_chunk_size,
    )
    total_sections = len(sections)
    if MAX_SECTIONS_FOR_DEMO is not None and total_sections > MAX_SECTIONS_FOR_DEMO:
        limit = MAX_SECTIONS_FOR_DEMO
        print(f"     → 共 {total_sections} 段, Demo 截取前 {limit} 段")
        sections = sections[:MAX_SECTIONS_FOR_DEMO]
    else:
        print(f"     → {total_sections} 段")

    if verbose:
        # 详细模式：带边框的完整内容
        for i, sec in enumerate(sections):
            heading = " > ".join(sec.heading_chain) if sec.heading_chain else "无标题"
            _box(
                f"段落 {i + 1}/{len(sections)} 「{heading}」 {len(sec.content)}字符",
                sec.content.splitlines(),
            )
    else:
        # 简洁模式：一行一段，截断预览
        for i, sec in enumerate(sections):
            heading = " > ".join(sec.heading_chain) if sec.heading_chain else "无标题"
            preview = _truncate(sec.content.replace("\n", " "))
            print(f"     {i + 1:>2}. [{heading}] {len(sec.content):>4}字符 │ {preview}")

    _done("[2/5] 语义切分", phase)

    # ③ 实体抽取
    phase = _phase("[3/5] 实体抽取 (LLM)")
    entity_extractor = EntityExtractor(
        llm_client,
        extraction_config=extraction_cfg,
    )
    entities = await entity_extractor.extract(sections)
    print(f"     → {len(entities)} 个实体")

    ent_items = [f"{e.name}({e.entity_type})" for e in entities]
    limit = 0 if verbose else 15
    for row in _compact_rows(ent_items, max_items=limit):
        print(f"     {row}")

    _done("[3/5] 实体抽取", phase)

    # ④ 三元组抽取
    phase = _phase("[4/5] 三元组抽取 (LLM)")
    triple_extractor = TripleExtractor(
        llm_client,
        extraction_config=extraction_cfg,
    )
    triples = await triple_extractor.extract(sections, entities)
    print(f"     → {len(triples)} 个三元组")

    display_triples = triples if verbose else triples[:15]
    for tri in display_triples:
        print(f"     {tri.subject} ─[{tri.predicate}]→ {tri.object}")
    if not verbose and len(triples) > 15:
        print(f"     … 共 {len(triples)} 个")

    _done("[4/5] 三元组抽取", phase)

    # ⑤ 知识图谱构建
    phase = _phase("[5/5] 构建知识图谱 (Neo4j)")
    await neo4j_client.execute("MATCH (n) DETACH DELETE n")
    print("     → 已清空旧图谱")

    builder = GraphBuilder(neo4j_client, auto_create_missing_nodes=True)
    stats = await builder.build_from_extraction(entities, triples)
    print(f"     → 节点: {stats.nodes_created}  关系: {stats.relations_created}")

    node_result = await neo4j_client.execute("MATCH (n) RETURN count(n) as cnt")
    rel_result = await neo4j_client.execute("MATCH ()-[r]->() RETURN count(r) as cnt")
    node_count = node_result[0]["cnt"] if node_result else 0
    rel_count = rel_result[0]["cnt"] if rel_result else 0
    _done("[5/5] 图谱构建", phase)

    elapsed = time.time() - t0
    print(f"\n  ✅ 离线完成 ({elapsed:.1f}s) — {node_count} 节点, {rel_count} 关系")

    return {"nodes": node_count, "relations": rel_count, "elapsed_s": int(elapsed)}


# ──────────────────── 在线阶段 ────────────────────


async def run_online_qa(
    question: str,
    neo4j_client: Neo4jClient,
    llm_client: BaseLLMClient,
    *,
    verbose: bool = False,
) -> None:
    """在线问答：问题 → 多跳推理 → 答案。"""
    print(f"\n  {'─' * (_W - 2)}")
    print(f"  ❓ {question}")
    print(f"  {'─' * (_W - 2)}")

    t0 = time.time()

    # ① 问题解析
    phase = _phase("[1/5] 解析问题意图")
    # Fetch known entity names from graph for fallback matching
    _name_rows = await neo4j_client.execute("MATCH (n) RETURN n.name AS name")
    known_entities = [r["name"] for r in _name_rows if r.get("name")]
    parser = QuestionParser(llm_client, known_entities=known_entities)
    parsed = await parser.parse(question)
    print(f"     意图: {parsed.intent.value}  实体: {parsed.entities}")
    if parsed.relation_hints:
        print(f"     关系提示: {parsed.relation_hints}")
    _done("[1/5] 问题解析", phase)

    # ② 查询重写 (QueryRewriter)
    query_plan = None
    schema_path = Path("config/domain_schema.json")
    if schema_path.exists():
        phase = _phase("[2/5] 查询重写 (QueryRewriter)")
        domain_schema = DomainSchema.load(schema_path)
        rewriter = QueryRewriter(llm_client, domain_schema=domain_schema)
        query_plan = await rewriter.rewrite(parsed)
        print(f"     起始实体: {query_plan.start_entities}")
        for i, step in enumerate(query_plan.steps):
            label_info = f" label={step.target_type}" if step.target_type else ""
            rel_info = f" rel={step.relation_hint}" if step.relation_hint else ""
            print(
                f"     步骤 {i + 1}: {step.action}"
                f" dir={step.direction}{label_info}{rel_info}"
                f"  ({step.description})"
            )
        _done("[2/5] 查询重写", phase)
    else:
        print("  ⏭️  跳过查询重写（无 domain_schema.json）")

    # ③ 多跳推理
    phase = _phase("[3/5] 多跳推理 (图谱检索)")
    retriever = GraphRetriever(neo4j_client)
    orchestrator = ReasoningOrchestrator(retriever, llm_client)
    evidence = await orchestrator.reason(parsed, query_plan=query_plan)

    _box("推理路径", _format_evidence(evidence, verbose=verbose))

    if verbose:
        # 详细模式额外显示完整 XML
        print("\n     📋 XML 证据链:")
        for xml_line in evidence.to_xml().splitlines():
            print(f"       {xml_line}")

    _done("[3/5] 多跳推理", phase)

    # ④ 上下文组装
    phase = _phase("[4/5] 组装推理上下文")
    assembler = ContextAssembler()
    context = assembler.assemble(question, evidence, include_reasoning=True)
    print(f"     → {len(context.prompt):,} 字符, {len(context.reasoning_steps)} 步推理")
    _done("[4/5] 上下文组装", phase)

    # ⑤ 答案生成
    phase = _phase("[5/5] 生成答案 (LLM)")
    generator = AnswerGenerator(llm_client)
    answer = await generator.generate(context, include_reasoning=True)
    _done("[5/5] 答案生成", phase)

    elapsed = time.time() - t0

    # 输出结果
    print(f"\n  {'═' * (_W - 2)}")
    print(f"  💡 回答  (置信度: {answer.confidence:.0%} · 耗时: {elapsed:.1f}s)")
    print(f"  {'═' * (_W - 2)}")
    for line in answer.answer.splitlines():
        print(f"  {line}")

    if verbose and answer.reasoning_steps:
        print("\n  📎 推理步骤:")
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
        default=0,
        help="最大处理段落数 (默认: 10, 0=全部)",
    )
    arg_parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细输出（完整段落内容、推理步骤、XML 证据链）",
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

    # ──── 日志归档 ────
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = log_dir / f"demo_{timestamp}.log"
    log_file = open(log_file_path, "w", encoding="utf-8")  # noqa: SIM115
    original_stdout = sys.stdout
    sys.stdout = _TeeStream(sys.stdout, log_file)  # type safe: TextIOBase subclass

    # 日志路由: file handler 始终 DEBUG, console handler 始终 INFO
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    # 加载配置
    config = load_config()

    # Verbose: root logger at DEBUG (file captures all), console stays INFO
    # Non-verbose: use config level for both
    if args.verbose:
        setup_logging("INFO")  # console handler = INFO
        logging.getLogger().setLevel(logging.DEBUG)  # root = DEBUG → file gets all
    else:
        setup_logging(config.logging.level)

    # ──── 启动横幅 ────
    print(f"{'═' * _W}")
    print("  🚀 多跳推理知识图谱问答系统 — Demo")
    print(f"{'═' * _W}")
    print(f"  模型: {config.llm.model_path}  Provider: {config.llm.provider}")
    print(f"  图DB: {config.graph.uri}")
    if args.verbose:
        print(f"  模式: verbose (调试日志 → {log_file_path})")

    # 初始化组件
    neo4j_client = Neo4jClient(config.graph)
    llm_client = create_llm_client(config.llm)

    try:
        # 连接 Neo4j
        print("\n  🔗 连接 Neo4j…")
        await neo4j_client.connect()
        print("     ✅ 已连接")

        # 启动 LLM
        print("  🤖 初始化 LLM…")
        await llm_client.start()
        print(f"     ✅ 已就绪 ({config.llm.provider})")

        # ═══════════════ 离线阶段 ═══════════════
        if not args.skip_offline:
            graph_stats = await run_offline_pipeline(
                args.doc,
                config,
                neo4j_client,
                llm_client,
                verbose=args.verbose,
            )
            if graph_stats["nodes"] == 0:
                print("\n  ⚠️  图谱为空，无法问答。请检查抽取结果。")
                return
        else:
            node_result = await neo4j_client.execute("MATCH (n) RETURN count(n) as cnt")
            node_count = node_result[0]["cnt"] if node_result else 0
            if node_count == 0:
                print("\n  ⚠️  图谱为空！请先运行离线阶段。")
                return
            print(f"\n  ⏭️  跳过离线（图谱已有 {node_count} 节点）")

        # ═══════════════ 在线阶段 ═══════════════
        print(f"\n{'═' * _W}")
        print("  🧠 在线阶段：多跳推理问答")
        print(f"{'═' * _W}")

        questions = [args.question] if args.question else DEFAULT_QUESTIONS

        for i, question in enumerate(questions):
            if len(questions) > 1:
                print(f"\n  ┌ 问题 {i + 1}/{len(questions)} ┐")
            await run_online_qa(
                question,
                neo4j_client,
                llm_client,
                verbose=args.verbose,
            )

        # ═══════════════ 交互模式 ═══════════════
        if not args.question:
            print(f"\n{'═' * _W}")
            print("  💬 交互模式（输入 quit 退出）")
            print(f"{'═' * _W}")

            while True:
                try:
                    user_input = input("\n  ❓ 你的问题: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n\n  👋 再见！")
                    break

                if not user_input:
                    continue
                if user_input.lower() in ("quit", "exit", "q", "退出"):
                    print("\n  👋 再见！")
                    break

                await run_online_qa(
                    user_input,
                    neo4j_client,
                    llm_client,
                    verbose=args.verbose,
                )

    finally:
        print("\n  🧹 清理资源…")
        await llm_client.stop()
        await neo4j_client.close()
        print("     ✅ 已释放")

        # 关闭日志归档
        sys.stdout = original_stdout
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()
        log_file.close()
        print(f"  📝 日志: {log_file_path}")


if __name__ == "__main__":
    asyncio.run(main())
