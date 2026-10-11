# 私有 GitHub 仓库

远端：[KDD](https://github.com/wangxiaoyunmaker/KDD)

此目录关联 origin，使用 main 分支。后续更新：

```bash
git status --short
git add .
git diff --cached --stat
git commit -m "Update research archive"
git push origin main
```

包含原始问卷、身份映射、人工标注和逐人研究产物，保持仓库 Private。密钥和环境目录被排除，`.env.example` 仅为空白模板。
