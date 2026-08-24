"""质询报告存储模块。

质询报告存储在 _inquiries.json 文件中，结构：
{
  "reports": [
    {
      "id": "inq_001",
      "timestamp": "2026-07-25T...",
      "question": "用户问题",
      "libraries": ["郎溪县志", "XX政策"],
      "inquiry_rounds": 2,
      "trigger_types": ["描述不一致", "重要性结论"],
      "errors": [...],
      "inconsistencies": [...],
      "findings_count": 3
    }
  ]
}
"""
from __future__ import annotations

import json
import os
from typing import List, Dict, Optional


class InquiryStore:
    """质询报告存储。"""

    def __init__(self, base_dir: str):
        self.base_dir = os.path.abspath(base_dir)
        self.store_path = os.path.join(self.base_dir, "_inquiries.json")
        self._data: Optional[Dict] = None

    def _load(self) -> Dict:
        if self._data is not None:
            return self._data
        if not os.path.isfile(self.store_path):
            self._data = {"reports": []}
        else:
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._data = {"reports": []}
        return self._data

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        tmp = self.store_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.store_path)

    def list_all(self) -> List[Dict]:
        """列出所有报告（按时间倒序）。"""
        data = self._load()
        reports = data.get("reports", [])
        # 按时间倒序
        reports.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
        return reports

    def get(self, report_id: str) -> Optional[Dict]:
        """按 ID 获取报告。"""
        data = self._load()
        for r in data.get("reports", []):
            if r.get("id") == report_id:
                return r
        return None

    def save(self, report: Dict) -> str:
        """保存报告，返回报告 ID。"""
        data = self._load()
        # 同 ID 覆盖
        reports = data.setdefault("reports", [])
        for i, r in enumerate(reports):
            if r.get("id") == report.get("id"):
                reports[i] = report
                self._save()
                return report.get("id", "")
        reports.append(report)
        self._save()
        return report.get("id", "")

    def delete(self, report_id: str) -> bool:
        """删除报告，返回是否成功。"""
        data = self._load()
        reports = data.get("reports", [])
        for i, r in enumerate(reports):
            if r.get("id") == report_id:
                reports.pop(i)
                self._save()
                return True
        return False

    def clear_all(self) -> int:
        """清空所有报告，返回删除数量。"""
        data = self._load()
        count = len(data.get("reports", []))
        data["reports"] = []
        self._save()
        return count
