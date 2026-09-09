# Methodology

本工具包使用一条可检查的程序化数据链路：

```text
scene program
→ deterministic solver
→ renderer
→ question template
→ rationale generator
→ verifier
→ SFT / diagnostic / probe / contrast routing
```

场景程序先采样结构化参数，求解器在渲染前计算唯一标签，渲染器再把同一场景转换成图像。问题模板只引用任务需要的信息；默认解释尽量由与标签相同的规则生成。验证器从保存的场景参数重新求解，而不是只检查已有答案字段。

不同用途通过 `use` 字段隔离。默认加载器只读取 `sft`；失败模式和探针不会因为同处一个仓库而被自动混入训练。

可见三面的视觉形式受到公开视觉推理任务常见形式的启发，但本仓库不包含第三方评测集的原题或原图。发布版本使用自己的固定朝向定义、Material 六色色板和程序化场景。

本方法能够验证“数据是否符合本生成器定义”，不能单独证明模型获得了通用视觉推理能力。外部迁移需要另行设计、完整保存并报告训练与评测实验。
