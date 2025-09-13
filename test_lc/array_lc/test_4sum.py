import unittest as ut

class Solution:
    def fourSum(self, nums: list[int], target: int) -> list[list[int]]:
        nums = sorted(nums)
        N = len(nums)

        answers = []
    
        for i in range(N-3):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            

            ni = nums[i]
            for j in range(i+1, N - 2):
                # check with the backwards.
                # not with the forwards
                # so that we avoid "already done" ones 
                if j > i + 1 and nums[j] == nums[j - 1]:
                    continue

                nj = nums[j]
                k = j + 1
                l = N - 1
                while k < l:
                    items = [ni, nj, nums[k], nums[l]]

                    items_sum = sum(items)
                    diff = target - items_sum

                    if diff < 0:
                        l -= 1
                    elif diff > 0:
                        k += 1
                    else:
                        answers.append(items)
                        # here, we increase/decrease both k and l
                        # because if we only change one of them, it will not equal the target
                        l -= 1
                        k += 1
                        # and since duplicates are not allowed, we check the previous value 
                        # continue incrementing if duplicate
                        while l > k and nums[l] == nums[l+1]:
                            l -= 1
                        while l > k and nums[k] == nums[k-1]:
                            k += 1
                            
                        
        return answers
    
    def move(self, nums: list[int], start: int, end: int) -> int:
        direction = (end-start) // abs(end - start)
        start += direction

        for i in range(start, end+direction, direction):
            if (i + direction) >= len(nums):
                return end
            
            if nums[i] != nums[i + direction]:
                return i
        return end

class Test(ut.TestCase):
    
    def test(self) -> None:
        cases = [
            ([1,0,-1,0,-2,2], 0, [[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]),
            ([2,2,2,2,2], 8, [[2,2,2,2]]),
            ([-2,-1,-1,1,1,2,2], 0, [[-2,-1,1,2],[-1,-1,1,1]])
        ]
        solution = Solution()
        for i, (input, target, expected) in enumerate(cases):
            with self.subTest(case=i):
                output = solution.fourSum(input, target)
                self.assertEqual(len(output), len(expected))
                
                for output_row, expected_row in zip(output, expected):
                    self.assertSequenceEqual(output_row, expected_row)
