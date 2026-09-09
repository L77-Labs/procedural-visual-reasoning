# cube_fold

推荐 SFT 子集。900 个程序化正方体展开图，只含 **固定朝向的图内相邻三面**（`task=cube_visible_triplet`，`use=sft`）。

按 `scene_id` 锁 split（约 795 train / 105 test）。图在 `images/visible/`：纯色块展开图 + 四幅轴测图。选项在图内。默认 `rationale` 由折叠几何生成（不点名 A–D，避免抄字母）；不是教师模型写的。

对面颜色题在 `../cube_opposite/`，不进默认训练集。

自检：`python scripts/verify_pack.py`（四选一恰好一个真角）。
