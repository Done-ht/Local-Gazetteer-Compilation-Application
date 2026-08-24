# -*- coding: utf-8 -*-
"""讯飞云端 OCR 模式（从 ocr-web 移植）。

功能：上传 pdf / docx / 常见图片，调用讯飞云端 OCR，输出可搜索 PDF / DOCX / TXT。
- 用户系统与 paddle 模式共用（app/auth）
- 数据按用户隔离（data/<user_id>/）
- 断点续传 + 页级并发 + 网络预检
"""
