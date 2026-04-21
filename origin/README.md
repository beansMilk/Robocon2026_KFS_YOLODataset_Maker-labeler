# KFS 数据集生成脚本说明

本文档说明以下 6 个脚本的用途、输出内容以及它们之间的关系：

- `KFS_maker_multiple_classes.py`
- `KFS_maker_multiple_cubes.py`
- `KFS_maker_single_cube_sphere.py`
- `KFS_maker_single_cube_sphere_bg.py`
- `KFS_maker_single_cube_vec.py`
- `KFS_seg.py`

## 各脚本基本信息

| 脚本名 | seg | pose | 单目标 | 多目标 | 单类别 | 多类别 | 提供PnP真值 | 360°拍摄 | 相机范围内移动 | 随机图片背景 | 纯色背景 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `KFS_maker_multiple_classes.py` |  | ✓ |  | ✓ |  | ✓ |  |  | ✓ |  | ✓ |
| `KFS_maker_multiple_cubes.py` |  | ✓ |  | ✓ | ✓ |  |  |  | ✓ |  | ✓ |
| `KFS_maker_single_cube_sphere.py` |  | ✓ | ✓ |  |  | ✓ |  | ✓ |  |  | ✓ |
| `KFS_maker_single_cube_sphere_bg.py` |  | ✓ | ✓ |  | ✓ |  |  |  | ✓ | ✓ |  |
| `KFS_maker_single_cube_vec.py` |  | ✓ | ✓ |  |  | ✓ | ✓ |  | ✓ |  | ✓ |
| `KFS_seg.py` | ✓ |  | ✓ |  |  | ✓ |  |  | ✓ | ✓ |  |

这些脚本的共同目标，都是通过 `pygame + OpenGL` 渲染带纹理的 KFS 立方体，自动生成训练数据。主要差别在于：

- 一张图里是单个 KFS 还是多个 KFS
- 是单类别还是多类别
- 是否加入真实背景
- 输出是关键点标签、位姿向量，还是分割标签

## 1. 整体关系

可以把这几个脚本理解成从“基础合成”到“任务特化”的几条分支：

1. `KFS_maker_multiple_cubes.py`
   生成多目标、单类别的 YOLO-pose 风格数据。
2. `KFS_maker_multiple_classes.py`
   在上面的基础上，扩展为多目标、多类别。
3. `KFS_maker_single_cube_sphere.py`
   改为单目标生成，并把相机位置改成球面采样，更适合精细姿态数据。
4. `KFS_maker_single_cube_sphere_bg.py`
   在 `single_cube_sphere` 的基础上加入真实背景和 train/val 划分。
5. `KFS_maker_single_cube_vec.py`
   在单目标关键点标签之外，额外输出相机位姿对应的 `rvec/tvec`。
6. `KFS_seg.py`
   复用与 `single_cube_sphere_bg` 相近的渲染流程，但输出改为 YOLO-seg 所需的面轮廓分割标签。

## 2. 各脚本功能

### `KFS_maker_multiple_cubes.py`

功能：

- 在一张图中放置多个 KFS 立方体
- 每个立方体可以使用不同贴图
- 输出检测框 + 8 个顶点关键点 + 可见性标记

特点：

- 输出目录为 `multi_cube_dataset_g2`
- 一张图内固定放置 6 个立方体
- 标签中的类别恒为 `0`
- 适合做“单类别、多实例”的目标检测/关键点训练

标签格式：

```txt
class cx cy w h x1 y1 v1 x2 y2 v2 ... x8 y8 v8
```

### `KFS_maker_multiple_classes.py`

功能：

- 与 `KFS_maker_multiple_cubes.py` 类似，也是单张图中生成多个 KFS
- 但每种贴图被视为一个独立类别
- 输出多类别的 YOLO-pose 风格标签

特点：

- 输出目录同样是 `multi_cube_dataset_g2`
- 一张图内也会放置多个立方体
- 标签中的 `class` 来自贴图编号，而不是固定为 `0`
- 适合做“多类别、多实例”的识别与关键点训练

和 `KFS_maker_multiple_cubes.py` 的核心区别：

- `multiple_cubes`：不同贴图只是外观变化，类别不变
- `multiple_classes`：不同贴图直接对应不同类别

### `KFS_maker_single_cube_sphere.py`

功能：

- 每张图只生成 1 个 KFS
- 相机位置通过球坐标随机采样
- 使用更复杂的 shader 和光照随机化，生成单目标关键点数据

特点：

- 输出目录为 `mono_cube_dataset_sphere`
- 输出 `images/` 和 `labels/`
- 同时创建了 `vectors/` 目录，但当前脚本本身没有写入位姿向量文件
- 更像是后续几个单目标脚本的基础版本

适用场景：

- 用于生成更加可控的单目标姿态/关键点训练集
- 适合作为 `single_cube_sphere_bg` 和部分单目标分支的基础实现

### `KFS_maker_single_cube_sphere_bg.py`

功能：

- 在 `KFS_maker_single_cube_sphere.py` 的基础上，加入真实背景图
- 将输出划分为训练集和验证集
- 仍然输出单目标关键点标签

特点：

- 背景来自 `KFS_backgrounds`
- 输出目录为 `mono_cube_dataset_demo`
- 输出结构为：
  - `images/train`
  - `images/val`
  - `labels/train`
  - `labels/val`
- 标签类别固定为 `0`
- 顶点输出顺序被改成了 `[6, 4, 0, 3, 7, 5, 1, 2]`

这个脚本的定位：

- 用于更贴近真实场景的单目标关键点训练数据生成
- 如果下游模型把所有 KFS 都当作同一个类别，这个脚本比 `single_cube_sphere.py` 更实用

### `KFS_maker_single_cube_vec.py`

功能：

- 生成单目标关键点数据
- 额外根据相机的 `eye / center / up` 计算真实位姿
- 为每张图保存 `rvec` 和 `tvec`

特点：

- 输出目录为 `mono_cube_dataset_vec`
- 输出 `images/`、`labels/`、`vectors/`
- `labels/` 中仍然是 YOLO-pose 风格关键点标签
- `vectors/` 中保存与图片同名的位姿向量文件

向量文件内容：

- 第 1 行：`rvec`
- 第 2 行：`tvec`

这个脚本的定位：

- 用于 PnP、6D pose、相机位姿回归或姿态验证
- 是“关键点标签 + 几何真值”同时具备的版本

### `KFS_seg.py`

功能：

- 生成单目标 KFS 图像
- 使用可见面判断逻辑，提取当前视角下可见面的 2D 四边形顶点
- 输出 YOLO-seg 风格的分割标签，而不是关键点标签

特点：

- 背景来自 `KFS_backgrounds`
- 输出目录为 `mono_cube_dataset_demo`
- 输出为 train/val 划分
- 每个可见面会写成标签文件中的一行
- 只有当可见面数量不少于 2 个时才保存样本

标签含义：

- 每一行对应一个可见面
- 行内容是 `class_id + 面轮廓归一化坐标`
- 本质上是把立方体的可见面拆成多个分割实例

这个脚本的定位：

- 用于训练分割模型，尤其是需要识别 KFS 各可见面的场景
- 是与 `single_cube_sphere_bg` 平行的一条“分割任务”分支

## 3. 输入与输出的共性

### 共同输入

- KFS 贴图目录：`../textures/KFS`
- 背景图目录：`KFS_backgrounds` 或脚本中写死的对应绝对路径

### 共同渲染方式

- 基于 `pygame` 创建 OpenGL 窗口
- 默认渲染分辨率为 `640 x 640`
- 随机相机、随机光照、纹理映射、地面平面共同组成合成场景

### 共同筛选逻辑

- 大多数脚本都会先判断顶点可见性
- 当可见顶点太少时，当前姿态会被丢弃，不写入数据集

## 4. 选用建议

如果你的目标是：

- 单类别多目标关键点训练：用 `KFS_maker_multiple_cubes.py`
- 多类别多目标关键点训练：用 `KFS_maker_multiple_classes.py`
- 单目标关键点训练：用 `KFS_maker_single_cube_sphere.py`
- 单目标关键点训练，且希望更贴近真实背景：用 `KFS_maker_single_cube_sphere_bg.py`
- 单目标关键点训练，并且还需要真实姿态向量：用 `KFS_maker_single_cube_vec.py`
- 单目标分割训练：用 `KFS_seg.py`

## 5. 一句话总结

- `multiple_cubes` 和 `multiple_classes` 解决“多目标”问题
- `single_cube_sphere`、`single_cube_sphere_bg`、`single_cube_vec` 解决“单目标姿态/关键点”问题
- `KFS_seg.py` 解决“单目标分割”问题
- 其中 `single_cube_sphere_bg` 和 `KFS_seg.py` 最接近真实场景，`single_cube_vec` 则最适合做几何位姿相关任务
