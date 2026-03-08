from __future__ import annotations

import json
from pathlib import Path

from docx import Document


def write_docx(file_path: Path, paragraphs: list[str]) -> None:
    document = Document()
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    document.save(str(file_path))


def build_medical_paragraphs() -> list[str]:
    return [
        "医学基础强调疾病、症状、治疗与器官系统之间的因果链条。高血压被认为是心血管系统的长期负荷性疾病，持续升高的血压会损伤血管内皮，使动脉壁更容易沉积脂质和炎症细胞，从而形成动脉粥样硬化。动脉粥样硬化进一步狭窄冠状动脉，诱发冠心病与心绞痛，因此心血管系统既是靶器官也是病变发生的核心部位。",
        "当冠心病导致心肌缺血时，典型症状是胸闷、胸痛和活动后呼吸急促。临床上常用硝酸甘油扩张冠状动脉以缓解心绞痛，同时配合β受体阻滞剂降低心率和心肌耗氧量。为降低血管炎症和脂质沉积，会加用他汀类药物稳定斑块，这一治疗链条体现了“疾病→器官→症状→药物”的多跳关系。",
        "肾素-血管紧张素-醛固酮系统在高血压中扮演关键角色。肾脏感知灌注不足时释放肾素，促使血管紧张素Ⅱ升高，引起血管收缩与水钠潴留。ACEI类药物通过抑制血管紧张素Ⅱ生成来降低血压，并可保护肾脏微血管，减少蛋白尿。由此形成“肾脏→肾素→血管紧张素Ⅱ→血压→ACEI”的因果链。",
        "糖尿病属于内分泌系统疾病，核心问题是胰岛β细胞分泌的胰岛素不足或作用抵抗。胰岛素缺乏导致血糖升高，长期高血糖会损害视网膜微血管，引发糖尿病视网膜病变；同时也会损害肾小球，引起糖尿病肾病。治疗上既要使用胰岛素或二甲双胍控制血糖，又要通过ACEI或ARB保护肾脏，体现了跨系统、多器官的治疗思路。",
        "炎症反应与免疫系统密切相关。感染性疾病如肺炎常由细菌或病毒引起，肺部出现咳嗽、发热、影像学渗出等症状。抗生素可以抑制细菌繁殖，退热药可以降低体温，但在病毒感染时抗生素无效，需要依赖免疫系统的清除功能。这里呈现“病原体→肺部→症状→药物→免疫”的关联。",
        "消化系统疾病中，胃食管反流会导致胸骨后烧灼痛和反酸。胃酸刺激食管黏膜产生炎症，质子泵抑制剂可以减少胃酸分泌，缓解食管损伤。若长期反复刺激可能导致巴雷特食管，增加癌变风险，因此治疗不仅针对症状，也需要保护黏膜这一结构。",
        "神经系统与代谢异常也有联系。长期糖尿病可能出现周围神经病变，表现为四肢麻木和感觉减退。血糖控制不良会加重神经损伤，而维生素B族可辅助神经修复。此处形成“糖尿病→高血糖→神经病变→维生素B族”的多跳关联。",
        "血液系统方面，贫血导致组织供氧不足，表现为乏力、心悸和头晕。铁缺乏性贫血需要补充铁剂并改善饮食结构；若贫血由慢性肾病引起，则需要促红细胞生成素治疗。不同病因通过不同治疗路径，体现了病因学与治疗策略的映射关系。",
        "综上，医学知识不是孤立条目，而是围绕疾病、症状、器官系统与药物建立的网络。高血压、冠心病、糖尿病等常见慢病之间存在共同危险因素和相互影响机制，使得多跳推理能够穿透系统边界，从病因追溯到器官，再到症状与治疗。",
    ]


def build_database_paragraphs() -> list[str]:
    return [
        "数据库系统的核心目标是可靠地存储与检索数据。关系数据库管理系统以表、行、列为基本结构，通过模式定义实体与关系。规范化可以减少冗余，但也可能引入过多连接操作，因此在实际系统中会结合业务需求进行适度的反规范化设计。",
        "SQL是关系数据库的标准查询语言，查询语句经过解析、优化与执行计划生成后才会真正访问数据。优化器会依据统计信息选择索引或全表扫描，索引通常采用B+树结构，能够将范围查询与排序开销降到较低水平。由此形成“SQL→优化器→执行计划→索引→性能”的链式关系。",
        "事务用于保证一组操作的原子性和一致性。ACID属性中，原子性依赖日志回滚机制，一致性与约束条件相关，隔离性通过锁或MVCC实现，持久性依赖重做日志和崩溃恢复。不同隔离级别影响并发读写行为，例如可重复读能避免不可重复读，但可能产生幻读，需要间隙锁或MVCC来控制。",
        "MySQL的InnoDB引擎实现了聚簇索引与行级锁，适合OLTP场景；其重做日志和双写缓冲提高了崩溃恢复能力。PostgreSQL采用多版本并发控制，读操作不阻塞写操作，适合复杂查询与分析任务。两者都支持标准SQL，但在执行计划、索引类型与扩展能力上存在差异。",
        "MongoDB属于文档型数据库，数据以JSON风格的文档存储，支持灵活的模式与嵌套结构。它通过副本集实现高可用，通过分片实现水平扩展，但其事务能力相对关系数据库更晚完善。选择MongoDB通常是因为文档模型更贴近应用对象，而不是因为传统SQL性能不足。",
        "索引设计要结合查询模式。例如在订单系统中，若常按用户ID与时间范围检索订单，可以建立联合索引，并保证最左前缀匹配规则。若同时需要排序与过滤，覆盖索引可减少回表次数，提高查询效率。索引虽然提升查询，但会增加写入成本，因此要在读写负载之间权衡。",
        "在分布式场景中，事务一致性还涉及两阶段提交与分布式锁。为了提升吞吐量，系统可能采用最终一致性，通过消息队列进行异步补偿。此时应用逻辑要理解一致性与可用性的折衷，确保业务可接受的数据延迟。",
        "数据备份与恢复是可靠性的重要部分。逻辑备份可以跨版本迁移，物理备份更适合快速恢复。增量备份结合归档日志可实现按时间点恢复，保证持久性目标。备份策略与持久性要求之间形成“业务RPO/RTO→备份策略→恢复流程”的因果链。",
        "总体而言，数据库系统将数据模型、查询语言、索引结构与事务机制连接成一个体系，具体产品如MySQL、PostgreSQL与MongoDB体现了不同的工程权衡，为多跳推理提供了丰富的实体与关系。",
    ]


def build_energy_paragraphs() -> list[str]:
    return [
        "新能源系统以太阳能、风能和储能为核心组成。光伏组件将太阳辐射转换为直流电，逆变器负责将直流变为交流并执行最大功率点跟踪。逆变器与并网保护装置协同，确保电能质量满足电网要求，从而实现“太阳能→光伏组件→逆变器→并网”的能量链。",
        "风电系统依赖风轮与发电机。风轮叶片的变桨控制可以在大风时降低载荷，发电机通过变流器输出稳定频率的电能。风电场接入电网后需要参与无功调节，以维持电压稳定，体现了“风能→变桨控制→变流器→无功调节→电网稳定”的关系。",
        "储能系统常采用锂电池，磷酸铁锂电池因安全性高而广泛应用。电池管理系统监测电压、电流与温度，防止过充过放，并与逆变器协同实现充放电控制。储能不仅用于削峰填谷，还用于调频与应急备用，使电网可以吸收更高比例的间歇性可再生能源。",
        "在电网侧，新能源并网引入了功率波动。为了提高稳定性，调度系统会综合预测模型、负荷曲线与储能容量进行优化。若风电输出突然下降，储能可快速放电补偿，避免频率下跌；而在光伏过剩时，储能吸收电能，缓解弃光问题。",
        "微电网是新能源利用的重要场景。微电网包含分布式光伏、风电与储能，并通过能量管理系统协调运行。在孤网模式下，储能提供惯量支撑，维持电压与频率；在并网模式下，微电网可以参与需求响应，降低主网负担。",
        "高压直流输电适合长距离输送风电和光伏电力。直流线路损耗低、稳定性好，但需要换流站进行交直流转换。新能源基地通过直流外送，将偏远地区的风光资源输送到负荷中心，形成“新能源基地→直流输电→负荷中心→电网消纳”的链条。",
        "电动汽车也逐渐成为储能资源。通过车网互动技术，车辆电池可在低谷时充电，在峰值时回馈电网。该模式需要计量与价格机制支持，并依赖充电桩与调度平台的协调。",
        "因此，新能源系统不仅涉及单一发电技术，还包含储能、并网控制、调度优化与电网稳定机制，这些实体和关系共同构成多跳推理的基础。",
    ]


def build_qa_pairs() -> dict[str, dict[str, object]]:
    return {
        "medical_basics": {
            "document": "test_documents/medical_basics.docx",
            "questions": [
                {
                    "question": (
                        "高血压如何通过血管内皮损伤发展为冠心病，"
                        "并通常需要哪些药物干预？"
                    ),
                    "expected_entities": [
                        "高血压",
                        "血管内皮",
                        "动脉粥样硬化",
                        "冠心病",
                        "他汀",
                    ],
                    "expected_hops": 3,
                },
                {
                    "question": "糖尿病导致肾损伤后，为什么会推荐ACEI或ARB？",
                    "expected_entities": ["糖尿病", "肾脏", "蛋白尿", "ACEI"],
                    "expected_hops": 2,
                },
                {
                    "question": "心绞痛的常见症状是什么，常用哪类药物缓解？",
                    "expected_entities": ["心绞痛", "胸痛", "硝酸甘油"],
                    "expected_hops": 1,
                },
                {
                    "question": (
                        "为什么高血糖会引发周围神经病变，常见的辅助治疗是什么？"
                    ),
                    "expected_entities": ["高血糖", "周围神经病变", "维生素B族"],
                    "expected_hops": 2,
                },
            ],
        },
        "database_systems": {
            "document": "test_documents/database_systems.docx",
            "questions": [
                {
                    "question": "SQL查询如何通过优化器选择索引来提升性能？",
                    "expected_entities": ["SQL", "优化器", "执行计划", "索引"],
                    "expected_hops": 2,
                },
                {
                    "question": "ACID中的持久性依赖哪些机制，为什么与崩溃恢复相关？",
                    "expected_entities": ["持久性", "重做日志", "崩溃恢复"],
                    "expected_hops": 2,
                },
                {
                    "question": "InnoDB为何适合OLTP场景？",
                    "expected_entities": ["InnoDB", "行级锁", "OLTP"],
                    "expected_hops": 1,
                },
                {
                    "question": "MongoDB的副本集与分片分别解决了什么问题？",
                    "expected_entities": [
                        "MongoDB",
                        "副本集",
                        "高可用",
                        "分片",
                        "水平扩展",
                    ],
                    "expected_hops": 2,
                },
            ],
        },
        "renewable_energy": {
            "document": "test_documents/renewable_energy.docx",
            "questions": [
                {
                    "question": "光伏系统中逆变器如何与并网保护配合保证电能质量？",
                    "expected_entities": ["光伏组件", "逆变器", "并网保护", "电能质量"],
                    "expected_hops": 2,
                },
                {
                    "question": "风电输出波动时储能如何参与调频以稳定电网？",
                    "expected_entities": ["风电", "储能", "调频", "电网稳定"],
                    "expected_hops": 2,
                },
                {
                    "question": "磷酸铁锂电池为什么需要BMS监测？",
                    "expected_entities": ["磷酸铁锂", "电池管理系统", "过充过放"],
                    "expected_hops": 1,
                },
                {
                    "question": "新能源基地通过哪种输电方式向负荷中心送电？",
                    "expected_entities": ["新能源基地", "高压直流输电", "负荷中心"],
                    "expected_hops": 2,
                },
            ],
        },
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "test_documents"
    output_dir.mkdir(parents=True, exist_ok=True)

    medical_path = output_dir / "medical_basics.docx"
    database_path = output_dir / "database_systems.docx"
    energy_path = output_dir / "renewable_energy.docx"
    qa_path = output_dir / "test_qa_pairs.json"

    medical_paragraphs = build_medical_paragraphs()
    database_paragraphs = build_database_paragraphs()
    energy_paragraphs = build_energy_paragraphs()

    write_docx(medical_path, medical_paragraphs)
    write_docx(database_path, database_paragraphs)
    write_docx(energy_path, energy_paragraphs)

    qa_pairs = build_qa_pairs()
    qa_path.write_text(
        json.dumps(qa_pairs, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Generated: {medical_path}")
    print(f"Generated: {database_path}")
    print(f"Generated: {energy_path}")
    print(f"Generated: {qa_path}")


if __name__ == "__main__":
    main()
