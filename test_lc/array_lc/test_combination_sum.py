from typing import List
import unittest as ut

class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        answers = []
        # index, combination, total
        queue = [(0, [], 0)]
        while queue:
            i, combination, total = queue.pop(-1)
            if total == target:
                answers.append(combination.copy())
                continue

            if total > target or i >= len(candidates):
                continue

            child_combination = combination.copy()
            child_combination.append(candidates[i])
            queue.append((i, child_combination, total + candidates[i]))

            queue.append((i + 1, combination, total))
        
        return answers
