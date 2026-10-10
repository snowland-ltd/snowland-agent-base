# Copyright 2026 Snowland
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""IDF 加权词面召回策略。"""
from __future__ import annotations

from typing import List, Optional, Sequence, Dict

from snowland_agent_base.skill.base import SkillMatcher
from snowland_agent_base.skill.models import Skill
from snowland_agent_base.skill.parsing import _idf, extract_terms


class IdfSkillMatcher(SkillMatcher):
    """IDF 加权词面召回。

    在大规模 skill 仓库 (数百个) 中, 朴素词频打分会把 "AI"、"数据" 这类高频词的
    无关 skill 排到前面 (如查询「写邮件」却命中含 "AI" 的资讯 skill)。故按词的
    文档频率降权, 并给 name 命中更高权重。
    """

    def _build_index(self, skills: Sequence[Skill]) -> Dict[str, Any]:
        df: Dict[str, int] = {}
        terms: Dict[str, tuple] = {}
        for skill_obj in skills:
            name_terms = extract_terms(skill_obj.name)
            desc_terms = extract_terms(skill_obj.description)
            terms[skill_obj.name] = (name_terms, desc_terms)
            for term in name_terms | desc_terms:
                df[term] = df.get(term, 0) + 1
        return {"df": df, "terms": terms, "total": len(skills)}

    def match(
        self, skills: Sequence[Skill], text: str, limit: Optional[int] = None
    ) -> List[Skill]:
        skills = list(skills)
        if not text or not skills:
            return []

        index = self._build_index(skills)
        df = index["df"]
        total = index["total"]
        query_terms = extract_terms(text)
        if not query_terms:
            return []

        scored: List[tuple] = []
        for skill_obj in skills:
            name_terms, desc_terms = index["terms"].get(skill_obj.name, (set(), set()))
            score = 0.0
            for term in query_terms:
                # IDF: 越稀有的词权重越高
                weight = _idf(term, df, total)
                if term in name_terms:
                    score += weight * 2.0  # name 命中权重更高
                elif term in desc_terms:
                    score += weight
            if score > 0:
                scored.append((score, skill_obj.name, skill_obj))

        scored.sort(key=lambda item: (-item[0], item[1]))
        result = [s for _, _, s in scored]
        return result[:limit] if limit else result
