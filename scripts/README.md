# Scripts

## 稳定公共接口

- `generate_cube_fold_v3.py`：生成固定朝向可见三面记录。
- `generate_cube_fold.py`：生成对面颜色诊断记录，并提供折叠几何函数。
- `generate_vessels_v2.py`：生成连通器记录，并提供确定性模拟器。
- `load_sft.py`：加载推荐 SFT 子集。
- `verify_pack.py`：重算并检查当前数据包。
- `release_check.py`：发布前检查 Schema 契约、全局 ID、Manifest 和文本卫生；安装 `requirements-dev.txt` 后还会执行完整 JSON Schema 校验。
- `rerender_pack_images.py`：从已保存场景重新渲染图片。

三个生成器默认输出本仓库 `schema.json` 兼容的 `records.jsonl`，图片路径相对于输出目录。

## 维护工具

其余脚本用于构建、迁移或修复当前发布包，不属于稳定 API。尤其是 `export_opensource_pack.py`，它需要未随发行版提供的原始构建树；普通使用者不需要运行它。
