




# qzone_auto_poster

AstrBot 插件
这是一个可以让llm自主发送qq空间说说的插件，需要手动获得qq空间的cookie，详情请见readme或github项目下说明。
# 使用方法
将bot账号登陆qq空间，然后按F12打开开发者模式，选择network然后刷新，点击任意一个请求寻找cookie（如果没有，可以换一个请求再找）然后将cookie复制并填到配置里。
![alt text](image.png)
注意，由于qq空间登录机制，cookie有时候会过期（比如bot掉线或您手动刷新了qq空间网页时，以及其他一些自然过期情况）需要重新获取
# 其他
如果您想更改说说发送是否成功返回的信息，可以在main.py中修改。
如果您有更好的想法或改进的想法，请随意使用！不过请遵循Astrbot的插件手册和开源协议。
# 支持
Astrbot[帮助文档](https://astrbot.app)

