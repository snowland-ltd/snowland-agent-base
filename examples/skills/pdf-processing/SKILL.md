---
name: pdf-processing
description: 提取 PDF 文本与表格、填写表单、合并文件。当用户提到 PDF、表单、文档抽取时使用。
license: MIT
metadata:
  author: snowland
  version: "1.0"
---

## 适用场景

用户需要处理 PDF 文档时使用本 skill，包括：文本提取、表格抽取、表单填写、文件合并。

## 标准流程

1. 确认用户意图：提取文本 / 抽取表格 / 填写表单 / 合并文件。
2. 调用 `extract_pdf_text` 提取文本内容。
3. 若需抽取表格，读取 `references/TABLES.md` 中的表格结构说明后再处理。
4. 涉及多个文件时，先提取再合并，避免顺序错乱。

## 注意事项

- 扫描件（图片型 PDF）无文本层，需提示用户先做 OCR。
- 大文件建议分页处理，避免一次性加载导致超时。
- 表单字段以 `references/FORMS.md` 中的命名为准。

## 输出要求

- 提取结果按原文档页码顺序组织。
- 若部分页面提取失败，需明确告知失败的页码，不要静默跳过。
