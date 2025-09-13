import unittest as ut

class Solution:
    def fourSum(self, nums: list[int], target: int) -> list[list[int]]:
        nums = sorted(nums)
        N = len(nums)

        answers = []
    
        for i in range(N-3):
            

            ni = nums[i]
            for j in range(i+1, N - 2):
                nj = nums[j]
                k = j + 1
                l = N - 1
                while k < l:
                    items = [ni, nj, nums[k], nums[l]]

                    items_sum = sum(items)
                    diff = target - items_sum

                    if diff < 0:
                        l = self.move(nums, l, k)
                    elif diff > 0:
                        k = self.move(nums, k, l)
                    else:
                        k = self.move(nums, k, l)
                        answers.append(items)
        return answers
    
    def move(self, nums: list[int], start: int, end: int) -> int:
        direction = (end-start) // abs(end - start)
        start += direction

        for i in range(start, end, direction):
            if nums[i] != nums[i + direction]:
                return i
        return end

class Test(ut.TestCase):
    
    def test(self) -> None:
        cases = [
            ([1,0,-1,0,-2,2], 0, [[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]),
            ([2,2,2,2,2], 8, [[2,2,2,2]])
        ]
        solution = Solution()
        for i, (input, target, expected) in enumerate(cases):
            with self.subTest(case=i):
                output = solution.fourSum(input, target)
                self.assertEqual(len(output), len(expected))
                
                for output_row, expected_row in zip(output, expected):
                    self.assertSequenceEqual(output_row, expected_row)

if __name__ == "__main__":
    ut.main()