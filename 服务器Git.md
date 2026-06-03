# 服务器项目接入 Git 并配合 Codex/VS Code 使用

本文整理本次对话中关于“云端 Linux 服务器项目如何接入 Git，并在本地 Codex 桌面版中辅助修改代码”的操作流程与常见报错处理。

## 一、背景

当前情况：

- 项目文件夹在云端 Linux 服务器上。
- 项目原本没有 Git 仓库。
- 本地使用的是 Codex 桌面版。
- 希望让 Codex 辅助修改代码，并能和 VS Code 配合查看、编辑、提交变更。

推荐方案：

1. 先在服务器项目目录中初始化 Git。
2. 将服务器当前代码推送到 GitHub 仓库。
3. 在本地通过 Codex 克隆 GitHub 仓库。
4. 本地修改后提交并推送。
5. 服务器通过 `git pull` 获取更新。

## 二、在服务器项目目录初始化 Git

先 SSH 登录服务器，然后进入项目目录：

```bash
ssh 用户名@服务器IP
cd /服务器上的项目路径
```

例如本次项目目录为：

```bash
cd ~/APDP-RTFL
```

初始化 Git：

```bash
git init
git status
```

## 三、建议先配置 `.gitignore`

在首次提交前，建议添加 `.gitignore`，避免把依赖、缓存、日志、密钥文件提交到仓库。

常见内容示例：

```gitignore
node_modules/
vendor/
__pycache__/
*.pyc
.env
.env.*
*.log
logs/
dist/
build/
.cache/
.DS_Store
```

如果项目包含数据库密码、API Key、证书、服务器配置文件等，应确认它们已经被 `.gitignore` 排除。

## 四、提交服务器当前代码

执行：

```bash
git add .
git commit -m "initial import from server"
```

### 报错：Author identity unknown

如果出现：

```text
Author identity unknown
Please tell me who you are.
fatal: unable to auto-detect email address
```

说明服务器上还没有配置 Git 提交身份。

推荐只给当前项目配置：

```bash
git config user.name "Hao"
git config user.email "你的GitHub或Gitee邮箱"
git commit -m "initial import from server"
```

如果希望这台服务器上所有 Git 仓库都使用同一个身份，可以使用全局配置：

```bash
git config --global user.name "Hao"
git config --global user.email "你的GitHub或Gitee邮箱"
git commit -m "initial import from server"
```

如果使用 GitHub 且不想暴露真实邮箱，可以使用 GitHub noreply 邮箱。

配置后不需要重新执行 `git add .`，直接重新执行：

```bash
git commit -m "initial import from server"
```

## 五、连接 GitHub 远程仓库

先在 GitHub 新建一个空仓库。注意：

- 不要勾选自动生成 README。
- 不要勾选 `.gitignore`。
- 不要勾选 license。

然后在服务器项目目录中执行：

```bash
git remote add origin https://github.com/你的用户名/仓库名.git
git branch -M main
git push -u origin main
```

本次示例仓库为：

```bash
git remote add origin https://github.com/haojy2024-bot/APDP-RTFL.git
git branch -M main
git push -u origin main
```

### 报错：syntax error near unexpected token `(`

如果执行的是类似下面这种命令：

```bash
git remote add origin [haojy2024-bot/APDP-RTFL.git](https://github.com/haojy2024-bot/APDP-RTFL.git)
```

会报错：

```text
bash: syntax error near unexpected token `('
```

原因是这是 Markdown 链接格式，不能直接粘贴到 Linux 终端。

正确写法是只保留纯 URL：

```bash
git remote add origin https://github.com/haojy2024-bot/APDP-RTFL.git
```

如果已经添加过远程地址，可以查看：

```bash
git remote -v
```

如果需要修改远程地址：

```bash
git remote set-url origin https://github.com/haojy2024-bot/APDP-RTFL.git
```

### 报错：GnuTLS recv error (-110)

如果出现：

```text
fatal: unable to access 'https://github.com/...': GnuTLS recv error (-110): The TLS connection was non-properly terminated.
```

通常是云服务器访问 GitHub 的 HTTPS 连接不稳定。

可以先重试：

```bash
git push -u origin main
```

如果多次失败，可以改用 SSH 推送：

```bash
ssh-keygen -t ed25519 -C "你的邮箱"
cat ~/.ssh/id_ed25519.pub
```

把公钥内容添加到 GitHub：

```text
GitHub -> Settings -> SSH and GPG keys -> New SSH key
```

然后修改远程地址：

```bash
git remote set-url origin git@github.com:haojy2024-bot/APDP-RTFL.git
ssh -T git@github.com
git push -u origin main
```

如果 `ssh -T git@github.com` 显示认证成功，再执行 push 即可。

## 六、确认推送成功

如果看到类似输出：

```text
To https://github.com/haojy2024-bot/APDP-RTFL.git
 * [new branch]      main -> main
Branch 'main' set up to track remote branch 'main' from 'origin'.
```

说明服务器项目已经成功推送到 GitHub。

### 大文件 warning

本次出现过如下提醒：

```text
remote: warning: File APDP-RTFL/application_record.csv is 51.83 MB; this is larger than GitHub's recommended maximum file size of 50.00 MB
remote: warning: GH001: Large files detected. You may want to try Git Large File Storage
```

这不是失败，只是提醒该文件超过 GitHub 推荐的 50 MB 单文件大小。

GitHub 的硬限制通常是 100 MB，所以这次可以成功推送。

后续如果继续管理大文件，建议考虑：

- 将不需要版本管理的大数据文件加入 `.gitignore`。
- 或者使用 Git LFS 管理大文件。

本次涉及的大文件是：

```text
APDP-RTFL/application_record.csv
```

## 七、在本地 Codex 桌面版克隆项目

如果没有 PowerShell，也可以直接对 Codex 输入：

```text
请把这个 GitHub 仓库克隆到本地：
https://github.com/haojy2024-bot/APDP-RTFL.git

保存到：
C:\Users\Hao\Documents\Codex\APDP-RTFL
```

Codex 可以帮助执行：

```powershell
cd C:\Users\Hao\Documents\Codex
git clone https://github.com/haojy2024-bot/APDP-RTFL.git
```

克隆完成后，可以用 VS Code 打开：

```text
C:\Users\Hao\Documents\Codex\APDP-RTFL
```

也可以在 VS Code 中手动选择：

```text
File -> Open Folder -> C:\Users\Hao\Documents\Codex\APDP-RTFL
```

## 八、后续推荐工作流

### 本地修改代码

让 Codex 在本地仓库中修改代码，例如：

```text
我的项目在 C:\Users\Hao\Documents\Codex\APDP-RTFL，请先阅读项目结构，然后帮我修改 XXX 功能。
```

修改后在本地提交并推送：

```powershell
cd C:\Users\Hao\Documents\Codex\APDP-RTFL
git status
git add .
git commit -m "update project"
git push
```

### 服务器同步代码

服务器上进入项目目录：

```bash
cd ~/APDP-RTFL
git pull
```

这样服务器就能获得本地 Codex 修改并推送到 GitHub 的最新代码。

## 九、关键注意事项

1. 不要把 Markdown 链接格式直接粘贴到终端，终端只接受纯 URL。
2. 首次提交如果报 `Author identity unknown`，配置 `user.name` 和 `user.email` 后重新 commit。
3. `GnuTLS recv error (-110)` 多半是网络问题，可以重试，必要时改用 SSH。
4. 大文件 warning 不等于失败，但后续应考虑 `.gitignore` 或 Git LFS。
5. 服务器项目接入 Git 后，建议以后都通过 GitHub 在本地和服务器之间同步，不再手动来回传整个项目目录。
