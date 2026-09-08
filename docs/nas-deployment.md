# 飞牛 NAS：按学期自动归档

本部署在 Canvas 发布目标学期后自动读取课程清单，唯一匹配 iCourse
课程，并把摘要和完整转写写入同一课程目录的 `录课转写/` 子目录。
视频和音频仅通过管道处理，不会保存到磁盘。

## 持久化路径

- `./data/2026-fall`：本学期 SQLite 与课程映射缓存。
- `./models`：SenseVoice 与 Silero VAD 模型。
- `/vol3/1000/kniplab/26-27秋学期`：Canvas 课件及录课转写。
- Canvas 部署的 `.state/2026-fall-courses.json`：无凭据的课程清单，只读挂载。

## 凭据

Compose 通过 `STUID_FILE` 和 `UISPSW_FILE` 读取 Canvas 部署已经维护的
Docker secret。不要把真实凭据写入仓库。模型 API Key 与可选 SMTP 配置放在
未跟踪的 `.env.nas`，文件权限应为 `0600`。

## 启动

```sh
cp .env.nas.example .env.nas
chmod 600 .env.nas
docker compose up -d --build
```

默认在 Asia/Shanghai 时区每天 13:00 和 22:00 运行，并使用文件锁防止重叠，
失败时最多重试三次。目标 Canvas 清单尚未
生成、课程没有 iCourse 录播，或课程匹配不唯一时，任务会记录原因并安全跳过。
也可设置 `RUN_MODE=once` 后运行一次性容器进行预检。

## Canvas 尚未发布但 iCourse 已有录播

可在 NAS 私有数据目录保存用户明确确认的 iCourse 名单，并设置
`CONFIRMED_COURSES_PATH=/app/data/confirmed-courses.json`。
仅当 Canvas manifest 尚不存在时使用；不伪造 Canvas ID 或 Term ID。
名单须包含 `schema_version: 1`、与配置一致的 `term_id` 以及 `courses` 数组。
每项包含 `icourse_id`、`course_code`、`name`、`teachers`。
执行前核对课程名称和教师；仍只处理平台已开放回放的课次。
Canvas 清单生成后恢复原有 manifest 自动发现流程，沿用同一数据库去重。
