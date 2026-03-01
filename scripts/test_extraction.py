"""快速测试新抽取流程（不需要 Neo4j）。

只跑前4步：文档加载 → 语义切分 → 实体抽取 → 三元组抽取
用于验证改进后的抽取质量。

用法: uv run python scripts/test_extraction.py [--sections N]
"""

from __future__ import annotations

import argparse
import asyncio
import time

from src.common.config import load_config
from src.data_processing.document_loader import DocumentLoader
from src.data_processing.entity_extractor import EntityExtractor
from src.data_processing.triple_extractor import TripleExtractor
from src.llm.client_factory import create_llm_client


async def main(max_sections: int = 5) -> None:
    config = load_config()
    extraction_cfg = config.extraction

    # 1. 加载文档
    print("=" * 60)
    print("[1/4] 加载文档...")
    loader = DocumentLoader()
    doc = loader.load("test_documents/03.docx")
    print(f"  文档: {doc.source_path}")
    print(f"  大小: {len(doc.content)} 字符")

    # 2. 语义切分
    print(f"\n[2/4] 语义切分 (max_chunk_size={extraction_cfg.entity_chunk_size})...")
    sections = loader.split_into_sections(
        doc.content, max_chunk_size=extraction_cfg.entity_chunk_size
    )
    print(f"  共 {len(sections)} 个语义段落")
    for s in sections[:10]:
        heading = " > ".join(s.heading_chain) or "(无标题)"
        print(f"    [{s.index}] [{heading}] {len(s.content)} 字符")
    if len(sections) > 10:
        print(f"    ... 省略 {len(sections) - 10} 个段落")

    # 只取前 N 个段落进行测试
    test_sections = sections[:max_sections]
    print(f"\n  ⚡ 测试模式: 只处理前 {len(test_sections)} 个段落")

    # 3. 实体抽取
    print(f"\n[3/4] 实体抽取 (temperature={extraction_cfg.temperature})...")
    llm_client = create_llm_client(config.llm)
    await llm_client.start()

    entity_extractor = EntityExtractor(llm_client, extraction_config=extraction_cfg)
    t0 = time.time()
    entities = await entity_extractor.extract(test_sections)
    t_entity = time.time() - t0
    print(f"  抽取到 {len(entities)} 个实体 ({t_entity:.1f}s)")
    for e in entities:
        aliases_str = f" (别名: {', '.join(e.aliases)})" if e.aliases else ""
        print(f"    - {e.name} [{e.entity_type}]{aliases_str}")

    # 4. 三元组抽取
    print(f"\n[4/4] 三元组抽取 (关系类型: {extraction_cfg.relation_types})...")
    triple_extractor = TripleExtractor(llm_client, extraction_config=extraction_cfg)
    t0 = time.time()
    triples = await triple_extractor.extract(test_sections, entities)
    t_triple = time.time() - t0
    print(f"  抽取到 {len(triples)} 个三元组 ({t_triple:.1f}s)")
    for t in triples:
        print(f"    ({t.subject}) --[{t.predicate}]--> ({t.object})")

    # 5. 质量统计
    print("\n" + "=" * 60)
    print("质量统计:")
    print(f"  段落数: {len(test_sections)}")
    print(f"  实体数: {len(entities)}")
    print(f"  三元组数: {len(triples)}")
    print(f"  实体抽取耗时: {t_entity:.1f}s")
    print(f"  三元组抽取耗时: {t_triple:.1f}s")
    if triples:
        predicates = [t.predicate for t in triples]
        from collections import Counter

        pred_counts = Counter(predicates)
        print(f"  关系类型分布: {dict(pred_counts)}")

    # 检查自反关系
    self_loops = [t for t in triples if t.subject == t.object]
    if self_loops:
        print(f"  ⚠️  自反关系: {len(self_loops)} 个 (应为0)")
    else:
        print(f"  ✅ 自反关系: 0 个")

    await llm_client.stop()
    print("\n完成!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试新抽取流程")
    parser.add_argument(
        "--sections", type=int, default=5, help="测试的段落数 (默认: 5)"
    )
    args = parser.parse_args()
    asyncio.run(main(max_sections=args.sections))
