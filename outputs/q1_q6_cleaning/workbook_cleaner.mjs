import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const sourcePath = "/Users/wy/Desktop/副本task_responses_cleaned.xlsx";
const outputDir = "/Users/wy/Documents/designer_roleplay/outputs/q1_q6_cleaning";
const mode = process.argv[2] || "inspect";
const finalPath = path.join(outputDir, "task_responses_q1-q6_cleaned_final.xlsx");

const textEdits = {
  15: `老板的上线需求仍需满足。我能做的是在达成上线目标的基础上，为用户保留一些喘息空间；否则上线效果不好，最后仍可能被追责。`,
  26: `我的第一反应主要来自直觉：先在脑中形成产品和使用场景的画面。`,
  27: `我的设计判断主要依赖直觉：先整合零散观察形成核心逻辑，再拆解问题，并用愿景和既有经验加以校准。`,
  29: `我倾向凭直觉寻找兼顾隐私与商业需求的全新设计方案。`,
  30: `我暂时说不清具体理由，但更倾向把精力放在用户行为和业务痛点上。`,
  31: `我主要凭直觉判断，倾向按原方案执行上线。`,
  35: `我会先梳理思路，再按轻重缓急推进各项分析与执行工作。`,
  85: `这些问题处于灰色地带，说明边界具有弹性，难以用两条绝对伦理红线框定。试图用单一设计同时满足隐私与商业需求也过于理想，因为现实中往往要牺牲一部分利益。即使从真实场景调研用户何时对数据敏感，也事实上接受了对部分用户数据的使用。因此，我会先依据合规与商业数据明确边界。`,
  86: `我认为 AI 可以作为辅助工具，但它缺少对本土文化和特色的深入洞察，仅靠它很难产出经得起时间检验的优秀产品。核心概念仍需要设计师独立思考和打磨，再让 AI 承担后期延展。完全禁止 AI 也有一定道理，但在商业环境中往往难以实施。`,
  108: `我倾向让 MVP 先行，通过小规模测试验证两种方案。`,
  115: `我的思路是：界定问题—设立目标—参考各行业案例—提炼异同与规律—梳理系统及利益相关者—产生新想法—将新想法置入系统—整理并总结方案。

现有内容没有覆盖“产生新想法”这一环节，但它对我的设计过程非常重要。新想法往往具有偶发性，可能并无明确依据，也不依赖具体工具，反而容易在休息或闲聊时出现，常常是现有系统中从未有过的解决方案。`,
  138: `我会先查看用户抑郁和焦虑的具体数据，以及 EEG 传感器的精度报告，再据此判断方案。`,
  139: `我会先拆解复杂问题，再评估技术与工具边界；随后整合零散观察形成核心逻辑，并设定不可牺牲的原则底线。`,
  163: `这与我的思考逻辑一致：先明确问题，再评估自身价值与边界，寻找可借鉴的案例，并形成相应愿景。`,
  170: `我认为先构想产品和使用场景，是这套设计首先要建立的框架；其他问题都应在这一框架下逐步解决。`,
  178: `我会优先从人因设计出发，构想产品形态和真实使用场景。`,
  179: `我会先界定问题，再借鉴既有经验和数据，提炼核心逻辑并设定原则底线。`,
  180: `我会先制作低保真原型并进行小规模测试，再根据用户的真实购买意愿做决定。`,
  181: `我认为需要正视利益驱动，并通过新的交互架构兼顾隐私与商业需求。`,
  194: `我会先构想产品形态和使用场景，重点考虑它与手机 App 的自然协同。`,
  195: `我会先评估技术与工具边界，再用客观数据验证可行性；随后平衡各方利益，并提炼可复用的设计原则。`,
  196: `我会先制作低保真原型并进行小规模测试，再根据用户的真实购买意愿做决定。`,
  197: `我会回到真实使用场景，调研用户何时对数据敏感，并把抽象伦理争议拆解为具体需求。`,
  198: `我会利用 AI 提高出图效率，把人的精力投入用户行为研究和 AI 难以触及的业务痛点。`,
  199: `我会执行上线指令，按原方案全量推向市场，不改动现有核心机制。`,
  310: `我更看重效率。既然 AI 已能替代最初级的设计工作，只要文字内容达到标准，这套方案就有可行性。用户选择产品的主要原因是内容输出能显著减少思考时间，而审美因人而异，也难以满足所有人；用户拿到内容后还可能自行做视觉调整。因此，我会优先升级文字功能。`,
  332: `我会先看产品如何定义、目标是什么，再判断相应技术如何实现。关键不在于一味追求更强，而在于是否精准匹配需求。因此，我更倾向在实用与美观之间取得平衡，并从用户使用、产品构造和商业价值等角度判断哪些因素重要、哪些可以取舍。`,
  365: `我会让调研结果说话，先弄清用户在真实场景中的数据敏感点。`,
  375: `我不愿参与一个明知有问题却仍强行推出的产品，因此会拒绝执行并与管理层交涉。`,
  379: `首先明确项目的核心目标和真实问题，接着开展背景分析、桌面调研与市场分析，梳理现有项目和经验。在此基础上寻找创新点与突破口，并评估其是否贴合实际、技术上是否可行，再用数据加以验证。完成初步排序后，我会按方向分类整理用户痛点，形成具体想法，并针对每个方向开展进一步分析和技术研究。`,
  382: `我不会完全放弃界面设计。以目前的技术水平，AI 生成界面与设计师作品仍有差异。我会使用 AI 辅助设计，把重点放在用户行为逻辑上；同时，面对企业降本增效和用 AI 替代专业交互设计的趋势，我仍会感到焦虑。`,
  412: `我认为客观数据和科学验证更重要，因此会优先选择准确率更高的方案。`,
  443: `我会结合生活习惯和既有经验理解问题，再用数据验证并调整核心设计逻辑。`,
  469: `我会以法律合规要求为准，先明确产品能够做什么、不能做什么。`,
  511: `我其实更想拒绝上线并与管理层正面交涉，但这种做法较为理想化。并非每个人都有与老板或甲方交涉的机会，也不是每个人都能承受退出项目的代价，主创还需要对团队负责。相比把优化全部推迟到下一阶段，我更倾向先满足上线要求，同时通过设计细节为用户保留自主空间，以解决燃眉之急。`,
  518: `我是用排除法做出的判断。完全依赖 AI 生成界面的测试过于迎合短期效率，彻底搁置或限制 AI 又过于理想化，因此我更倾向利用 AI 提升效率，同时把精力转向用户行为和业务痛点。`,
  519: `这是一个折中的下策：先按时上线保住先发优势，再把用户体验优化纳入下一次迭代。目前没有更理想、也更可执行的办法。`,
  548: `我认为美学体验非常重要，不能只追求准确率。`,
  571: `我的排序主要来自直觉：先形成愿景，再评估技术边界、参考既有经验并拆解问题，同时平衡各方利益。`,
  591: `这种做法能在项目进度和用户体验之间取得平衡：表面满足上线要求，同时通过细节优化为用户保留一定自主空间。`,
  628: `设计师与工程师对“丑陋”或“可用”的判断阈值往往不同。79% 的准确率是否足以支撑核心效率提升，99% 的准确率带来的舒适度损失是否会让用户拒绝佩戴，都应由目标用户的选择偏好和支付意愿来验证。内部争论或依赖假设都可能走偏，而小规模市场测试可在较短时间内获得真实购买意愿数据，比内部报告更有说服力。用用户是否愿意买单来比较两种方案，既避免唯数据论或唯美学论，也能用实证终结争议。`,
  630: `AI 生成界面再快，也只是对已有风格和模式的高效复刻。它难以理解本土用户为何会对某种颜色、隐喻或交互节奏产生情感共鸣，也看不到焦虑、孤独、归属感等深层行为逻辑，而这些正是设计的价值壁垒。与其限制 AI 或与它比拼出图速度，不如把精力投入用户研究、行为洞察和情感化体验设计。这样既能回应降本增效的诉求，也能推动设计师从界面执行者转向策略研究者和体验架构者：让 AI 负责执行，让人定义正确的问题，并指导 AI 生成更有温度、更本土化的方案。`,
  634: `我会重点关注 EEG 传感器的精度及相关测试数据。`,
  638: `我倾向在关键设计环节禁用 AI，以保留独立判断和设计质量。`,
  711: `如果老板或甲方缺乏基本社会责任感，我认为没有必要继续合作。暗中通过设计细节对抗系统不仅执行成本高，也很难真正奏效；把人文关怀推迟到下一阶段，则可能让产品在首发时就输给更重视用户的竞品，并在舆论上处于劣势。这类原则性问题应在第一版产品中就明确。`,
  726: `如果是我，我会使用趁手的 AI 工具提高效率，因为我对这份工作没有那么强的热情；但从设计理想出发，我仍不喜欢缺乏“人味”的 AI 图像。`,
  734: `引入 AI 并不一定会真正减少工作量，但作为执行者，我会尽量选择更高效的方式。既然这是领导推动的工具，就需要在流程中充分使用并证明其价值。我会让 AI 生成多套界面，再通过 A/B 测试让用户数据决定方案，同时把节省下来的精力投入更需要人工判断的工作。`,
  746: `人不是机器，也有立场和情感。作为设计师之前，我首先是一个人，因此不会让自己的工作违背个人的道德标准和价值观。`,
  749: `用全新的设计架构同时解决隐私和商业矛盾看似完美，但过于理想化。设计师只是产品开发中的一环，设计并非万能。更现实的方式是先确立不可逾越的原则红线，再尽力平衡各项矛盾。`,
  794: `设计项目刚开题时，应先从产品本身出发，构想可能的使用场景，并据此判断需求和实现路径。数据、伦理和技术等问题会随着项目推进逐步出现，不会一开始就成为主要思考。`,
  798: `AI 的介入是难以阻挡的趋势，从大量方案中筛选最优结果，本质上是对传统多方案迭代的加速。我不会排斥这种工作流，关键是如何利用它推进项目。我会把 AI 生成的多套界面用于 A/B 测试，把它视为一种 Best-of-N 的筛选过程：一方面帮助 AI 理解设计目标，另一方面积累对用户偏好的认识。这样可以提高界面执行效率，把人的精力投入其他更需要判断的环节。`,
  805: `我会先控制自己能够控制的部分，尽量做到问心无愧。同时，也可以通过调研真实场景中的数据敏感点，把抽象的道德议题拆解为具体需求。`,
  819: `我会从具体问题入手，以小见大：先界定核心问题并用数据验证，再平衡各方利益，最终形成整体愿景。`,
  870: `我认为单纯把精力从界面转向用户行为的表述并不完整，AI 与人更适合循环协作。老板显然希望降低成本，但全流程由 AI 完成的方案未必优于人工设计。要从根本上解决问题，可以让设计师与 AI 协作产出多个版本，再与纯 AI 版本进行 A/B 测试，用市场结果证明人机协作的商业价值。AI 替代低技术含量岗位可能是大势所趋，因为企业更看重效率和收益。`,
  923: `我的思路接近 PDCA：先明确问题和利益优先级，再拆解、整合、验证并总结原则，形成持续迭代的闭环。`,
  948: `我认为两种方案都需要先经过小规模测试验证，再依据真实反馈决策。`,
  981: `我认为先依据法律和客观数据界定边界，是更理性、稳妥的做法。`,
  999: `这是一个折中方案：在守住项目基本盘的前提下，为用户体验保留长期优化的可能性，也避免采取过于极端的处理方式。`,
  1010: `在这款产品需要决策的多个场景中，用户的抑郁和焦虑状况会直接影响一部分人是否愿意使用产品。产品初期最重要的是获得真实用户，因此这项决策最为紧急。我需要了解这部分用户的具体情况，以推动后续改进。`,
  1021: `我会先拆解具体痛点，弄清用户在哪些真实场景中对数据敏感。`,
  1046: `我选择先进行 A/B 测试，因为这是成本较低且相对客观的判断方式，既能减少主观偏见，也能提供老板和团队都更容易接受的证据，为后续更深入的设计工作铺路。`,
  1054: `我原本一直倾向先独立构思核心概念，再让 AI 执行后期延展；但把精力转向用户行为逻辑和 AI 难以触及的业务痛点，也有一定道理。`,
  1061: `我没有特别复杂的理由，主要是以法规为准。我还会考虑在用户须知中明确写明信息收集范围，并承诺进行严格的隐私管理。`,
  1062: `我倾向把精力转向用户行为和业务痛点，这就是我的主要理由；我本身也不喜欢做视觉设计。`,
  1092: `我会把准确性放在第一位，美观度作为辅助。现有方案都不能完全表达这个判断，因此我选择了相对接近自己想法的做法。`,
  1126: `我只能把精力聚焦到 AI 暂时无法替代的领域。完全依赖 AI 生成界面并让大众审美决定结果并不理想，因为大众美育水平不一；彻底搁置 AI 又过于一刀切。更合理的是利用 AI 提高执行效率，同时保留设计师在用户洞察和审美判断上的作用。`,
  1133: `我认为查阅法律与商业数据，和调研真实场景中的数据敏感点有相似之处。综合来看，我更倾向先用数据合规法案、竞品隐私协议和商业变现报告界定边界。产品要进入市场并实现盈利，首先应尊重法律和市场规则；违反规则最终会面临处罚和声誉损失。`,
  1149: `这种做法把抽象道德问题转化为可供设计师分析和优化的具体数据问题，也确实从用户角度出发。用户未必完全不能接受数据开放，关键是把握边界。因此，找到用户真正关切的痛点并有针对性地解决，会更有效。`,
  1157: `脑电波数据属于敏感个人信息，违规收集或使用不仅可能面临高额罚款，还可能导致产品下架和企业声誉受损。因此，首先应查阅合规法案和竞品隐私协议，明确法律允许的边界，避免项目直接触碰红线。

面对商业诉求和伦理争议，空泛的道德讨论难以解决问题。还需要调研商业变现报告和行业案例，用客观数据说明合规使用数据是否同样能实现商业价值，以及违规变现的风险成本是否远大于收益。

合规调研不是终点，而是明确“能做什么、不能做什么”的起点。在法律框架内，仍可探索本地化隐私计算、用户分级授权等方案，兼顾商业需求和用户隐私。`,
  1158: `AI 能高效生成标准化界面，却难以真正理解用户的本土文化、情绪需求和深层业务痛点，例如用户为何会在界面中感到焦虑、什么样的交互更有“人味”。因此，我会把精力转向用户行为逻辑和业务洞察，减少低价值的重复绘图。

老板希望通过 AI 降本增效，甚至淘汰初级岗位，本质上是因为界面绘制和风格套用等工作正在被替代。与其回避、限制或被动依赖 AI，不如主动调整工作重心，从界面执行者转向用户体验策略者：利用 AI 的效率优势，同时用更高层次的判断证明设计师的不可替代性。`,
  1159: `老板要求三个月内上线以抢占市场先机，这是当前的核心商业目标。按计划上线可以保住先发优势，避免因正面冲突或消极对抗导致项目停滞、团队被动。

但完全砍掉人性化设计，会让用户焦虑和异化感在上线后集中爆发，损害品牌口碑和长期留存。因此，我会把焦虑缓解和边缘用户优化明确纳入第二阶段迭代，既不放弃对用户体验的责任，也为后续修复和完善留下路径与时间。`,
  1183: `我会灵活应对上线压力，在满足项目进度的同时，通过细节优化为用户保留一定自主空间。`,
  1189: `我认为需要在隐私、商业目标和真实用户需求之间做出取舍。`,
  1205: `我会以用户为导向，调研真实场景中的数据敏感点，再拆解并解决具体痛点。`,
  1244: `我认为先制作原型并进行小规模测试，是验证两种方案可行性的合适方式。`,
  1266: `我会坚持以人为本，反对为了效率把人异化为机器。`,
  1267: `我的排序主要来自直觉，但会以核心问题为起点，用数据和技术评估校准判断，设定底线并平衡各方利益。`,
  1268: `这个取舍很难，但我更倾向依据客观性能数据选择准确率更高的方案。`,
  1269: `我暂时说不清具体理由，但会优先依据法律合规要求和客观商业数据界定边界。`,
  1274: `我对用户可能出现的抑郁和焦虑感同身受，因此希望先了解具体数据并据此改进。`,
};

await fs.mkdir(outputDir, { recursive: true });
const input = await FileBlob.load(mode === "verify" ? finalPath : sourcePath);
const workbook = await SpreadsheetFile.importXlsx(input);

if (mode === "inspect") {
  const overview = await workbook.inspect({
    kind: "workbook,sheet,table",
    maxChars: 12000,
    tableMaxRows: 8,
    tableMaxCols: 12,
    tableMaxCellChars: 160,
  });
  console.log(overview.ndjson);

  const sheet = workbook.worksheets.getItem("task_responses_cleaned");
  const dataRange = sheet.getRange("A1:G1297");
  await fs.writeFile(
    path.join(outputDir, "source_values.json"),
    JSON.stringify(dataRange.values, null, 2),
  );
  await fs.writeFile(
    path.join(outputDir, "source_formulas.json"),
    JSON.stringify(sheet.getRange("A1:A20").formulas, null, 2),
  );

  for (const [label, range] of [["top", "A1:G28"], ["bottom", "A1270:G1297"]]) {
    const preview = await workbook.render({
      sheetName: sheet.name,
      range,
      scale: 1,
      format: "png",
    });
    await fs.writeFile(
      path.join(outputDir, `before_${label}.png`),
      new Uint8Array(await preview.arrayBuffer()),
    );
  }
}

if (mode === "analyze") {
  const sheet = workbook.worksheets.getItem("task_responses_cleaned");
  const rows = sheet.getRange("A1:G1297").values;
  const headers = rows[0];
  const records = rows.slice(1).map((row, index) => ({
    excelRow: index + 2,
    ...Object.fromEntries(headers.map((header, col) => [header, row[col] ?? null])),
  }));

  const q1q6 = records.filter((record) => /^Q[1-6]$/.test(String(record.question_id)));
  const optionPattern = /(?:^|[\s，。；、（(「“"])(?:第[一二三四五六123456]+(?:个|项)?|选项\s*[A-FＡ-Ｆ]|[A-FＡ-Ｆ](?:\s*[、,，和或/＋+&]\s*[A-FＡ-Ｆ])+|[A-FＡ-Ｆ]\s*(?:项|选项)|选择?\s*[A-FＡ-Ｆ])(?:$|[\s，。；、）)」”"])/i;
  const metaPattern = /^(?:同上|如上|见上|都可以|无|暂无|没有|没什么|不清楚|不知道|不确定|其他|略|none|n\/a|na)$/i;
  const awkwardPattern = /(?:\.\.\.|……{2,}|，，|。。|；；|、、|\?\?|！！|的的|了了|是是|我我|因为.{0,20}因为|所以.{0,20}所以)/;

  const issues = q1q6.flatMap((record) => {
    const text = record.text_supplement == null ? "" : String(record.text_supplement).trim();
    const flags = [];
    if (!text) flags.push("blank");
    if (text && optionPattern.test(text)) flags.push("option_reference");
    if (text && metaPattern.test(text)) flags.push("non_substantive");
    if (text && text.replace(/\s/g, "").length <= 5) flags.push("very_short");
    if (text && awkwardPattern.test(text)) flags.push("surface_disfluency");
    if (/^[\s,，.。;；:：!?！？、\-—_]+$/.test(text)) flags.push("punctuation_only");
    return flags.length ? [{
      excelRow: record.excelRow,
      question_id: record.question_id,
      answer_value: record.answer_value,
      text_supplement: record.text_supplement,
      flags,
    }] : [];
  });

  const stats = {
    totalRows: records.length,
    q1q6Rows: q1q6.length,
    perQuestion: Object.fromEntries(
      [1, 2, 3, 4, 5, 6].map((q) => {
        const subset = q1q6.filter((record) => record.question_id === `Q${q}`);
        return [`Q${q}`, {
          rows: subset.length,
          blanks: subset.filter((record) => record.text_supplement == null || String(record.text_supplement).trim() === "").length,
        }];
      }),
    ),
    issueCounts: issues.reduce((acc, issue) => {
      for (const flag of issue.flags) acc[flag] = (acc[flag] || 0) + 1;
      return acc;
    }, {}),
    candidateRows: issues.length,
  };

  await fs.writeFile(path.join(outputDir, "analysis_stats.json"), JSON.stringify(stats, null, 2));
  await fs.writeFile(path.join(outputDir, "issue_candidates.json"), JSON.stringify(issues, null, 2));
  console.log(JSON.stringify(stats, null, 2));
}

if (mode === "references") {
  const sheet = workbook.worksheets.getItem("task_responses_cleaned");
  const rows = sheet.getRange("A1:G1297").values;
  const referencePattern = /(?:选项|选择(?:了|的是|为)?\s*[A-FＡ-Ｆ]|第[一二三四五六七八九十0-9]+(?:项|个|种|情景|场景)|(?:^|[^A-Za-z])A\s*[、,，和或/＋+&]\s*B(?:[^A-Za-z]|$)|(?:^|[\s，。；：、（(「“])(?:A|B|C|D|E|F)(?:[\s，。；：、）)」”]|$))/i;
  const hits = rows.slice(1).flatMap((row, index) => {
    const questionId = String(row[1] ?? "");
    const textSupplement = row[3] == null ? "" : String(row[3]);
    if (!/^Q[1-6]$/.test(questionId) || !referencePattern.test(textSupplement)) return [];
    return [{
      excelRow: index + 2,
      question_id: questionId,
      answer_value: row[2] ?? null,
      text_supplement: row[3] ?? null,
    }];
  });
  await fs.writeFile(path.join(outputDir, "reference_candidates.json"), JSON.stringify(hits, null, 2));
  console.log(JSON.stringify({ referenceCandidates: hits.length }, null, 2));
}

if (mode === "formula-audit") {
  const sheet = workbook.worksheets.getItem("task_responses_cleaned");
  const formulas = sheet.getRange("A2:A1297").formulas;
  const values = sheet.getRange("A2:A1297").values;
  const hits = formulas.flatMap((row, index) => row[0] ? [{
    excelRow: index + 2,
    formula: row[0],
    displayedValue: values[index][0],
  }] : []);
  await fs.writeFile(path.join(outputDir, "user_id_formula_audit.json"), JSON.stringify(hits, null, 2));
  console.log(JSON.stringify({ formulaUserIds: hits.length, hits }, null, 2));
}

if (mode === "clean") {
  const sheet = workbook.worksheets.getItem("task_responses_cleaned");

  for (const [excelRow, replacement] of Object.entries(textEdits)) {
    sheet.getRange(`D${excelRow}`).values = [[replacement]];
  }

  const idFormulas = sheet.getRange("A2:A1297").formulas;
  for (let index = 0; index < idFormulas.length; index += 1) {
    const formula = idFormulas[index][0];
    if (typeof formula === "string" && /^=-[A-Za-z0-9_-]+$/.test(formula)) {
      sheet.getRange(`A${index + 2}`).values = [[formula.slice(1)]];
    }
  }

  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(finalPath);
  await fs.writeFile(
    path.join(outputDir, "applied_edits.json"),
    JSON.stringify({ textEditRows: Object.keys(textEdits).map(Number), fixedFormulaUserIds: 16 }, null, 2),
  );
  console.log(JSON.stringify({ finalPath, textEdits: Object.keys(textEdits).length, fixedFormulaUserIds: 16 }, null, 2));
}

if (mode === "verify") {
  const sheet = workbook.worksheets.getItem("task_responses_cleaned");
  const rows = sheet.getRange("A1:G1297").values;
  const q1q6 = rows.slice(1).map((row, index) => ({ excelRow: index + 2, row }))
    .filter(({ row }) => /^Q[1-6]$/.test(String(row[1])));

  const blankRows = q1q6
    .filter(({ row }) => row[3] == null || String(row[3]).trim() === "")
    .map(({ excelRow }) => excelRow);
  const optionReferencePattern = /(?:选项|选择(?:了|的是|为)?\s*[A-FＡ-Ｆ]|第[一二三四五六七八九十0-9]+(?:项|个(?:选项|情景|场景)|种(?:选项|方案)|情景|场景)|(?:^|[^A-Za-z])A\s*[、，和或＋+&]\s*B(?:[^A-Za-z]|$)|(?:^|[\s，。；：、（(「“])(?:A|B|C|D|E|F)\s*(?:项|选项)(?:[\s，。；：、）)」”]|$))/i;
  const optionReferenceRows = q1q6
    .filter(({ row }) => optionReferencePattern.test(String(row[3] ?? "")))
    .map(({ excelRow }) => excelRow);
  const corruptionPattern = /(?:」\s*\/\s*「|一个但是|e\s* \s*e\s* \s*g|！！！！|，，|。。)/i;
  const corruptionRows = q1q6
    .filter(({ row }) => corruptionPattern.test(String(row[3] ?? "")))
    .map(({ excelRow }) => excelRow);
  const editMismatches = Object.entries(textEdits)
    .filter(([excelRow, expected]) => String(sheet.getRange(`D${excelRow}`).values[0][0] ?? "") !== expected)
    .map(([excelRow]) => Number(excelRow));

  const formulaErrors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 300 },
    summary: "final formula error scan",
    maxChars: 12000,
  });
  const keyRanges = await workbook.inspect({
    kind: "table",
    sheetId: sheet.name,
    range: "A138:D199",
    include: "values,formulas",
    tableMaxRows: 70,
    tableMaxCols: 4,
    tableMaxCellChars: 220,
    maxChars: 18000,
  });

  const previews = [["top", "A1:G28"], ["middle", "A620:G640"], ["bottom", "A1270:G1297"]];
  for (const [label, range] of previews) {
    const preview = await workbook.render({
      sheetName: sheet.name,
      range,
      scale: 1,
      format: "png",
    });
    await fs.writeFile(
      path.join(outputDir, `after_${label}.png`),
      new Uint8Array(await preview.arrayBuffer()),
    );
  }

  const report = {
    q1q6Rows: q1q6.length,
    blankRows,
    optionReferenceRows,
    corruptionRows,
    editMismatches,
    restoredUserIds: sheet.getRange("A2:A17").values.flat(),
    formulaErrorScan: formulaErrors.ndjson,
    keyRangeInspection: keyRanges.ndjson,
  };
  await fs.writeFile(path.join(outputDir, "verification_report.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify({
    q1q6Rows: report.q1q6Rows,
    blankRows: report.blankRows,
    optionReferenceRows: report.optionReferenceRows,
    corruptionRows: report.corruptionRows,
    editMismatches: report.editMismatches,
    restoredUserIds: report.restoredUserIds,
    formulaErrorScan: report.formulaErrorScan,
  }, null, 2));
}
