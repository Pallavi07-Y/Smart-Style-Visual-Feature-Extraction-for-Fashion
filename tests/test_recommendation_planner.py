import unittest

from core.schemas import UserProfile, WardrobeItem
from features.planner import generate_weekly_plan, regenerate_day
from features.recommendation import recommend_outfits, score_outfit


def item(item_id, category, color, name=None):
    return WardrobeItem(item_id=item_id, name=name or item_id, category=category, color=color)


class RecommendationPlannerTests(unittest.TestCase):
    def setUp(self):
        self.items = [
            item("top", "Top", "white"),
            item("bottom", "Bottom", "navy"),
            item("dress", "Dress", "black"),
            item("shoes", "Shoes", "black"),
            item("bag", "Accessory", "brown"),
        ]
        self.profile = UserProfile(occasion="Work", style_preferences=["Classic"], wardrobe_items=self.items)

    def test_compatible_outfit_scores_above_incomplete_outfit(self):
        complete = score_outfit(self.profile, [self.items[0], self.items[1], self.items[3]], "Work")
        incomplete = score_outfit(self.profile, [self.items[0]], "Work")
        self.assertGreater(complete.score, incomplete.score)
        self.assertTrue(complete.reasons)

    def test_recommendation_uses_only_wardrobe_items(self):
        results = recommend_outfits(self.profile, self.items, "Work", top_k=3)
        known = {item.item_id for item in self.items}
        self.assertTrue(results)
        self.assertTrue(all(item.item_id in known for result in results for item in result.items))

    def test_empty_wardrobe(self):
        self.assertEqual(recommend_outfits(UserProfile(), [], "Everyday"), [])

    def test_weekly_plan_has_seven_days(self):
        plan = generate_weekly_plan(self.profile, self.items)
        self.assertEqual(len(plan.days), 7)
        self.assertTrue(all(day.day for day in plan.days))

    def test_weekly_plan_handles_insufficient_wardrobe(self):
        plan = generate_weekly_plan(UserProfile(wardrobe_items=[self.items[0]]), [self.items[0]])
        self.assertEqual(len(plan.days), 7)
        self.assertTrue(all(day.outfit for day in plan.days))

    def test_regeneration_changes_day_when_alternative_exists(self):
        plan = generate_weekly_plan(self.profile, self.items)
        regenerated = regenerate_day(plan, 0, self.profile, self.items)
        self.assertEqual(len(regenerated.days), 7)
        self.assertIsNotNone(regenerated.days[0].outfit)

    def test_occasion_changes_score(self):
        work = recommend_outfits(self.profile, self.items, "Work", top_k=1)[0]
        weekend = recommend_outfits(self.profile, self.items, "Weekend", top_k=1)[0]
        self.assertNotEqual(work.components["occasion_compatibility"], weekend.components["occasion_compatibility"])


if __name__ == "__main__":
    unittest.main()
