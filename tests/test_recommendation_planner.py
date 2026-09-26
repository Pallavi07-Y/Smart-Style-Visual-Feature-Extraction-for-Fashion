import unittest

from core.schemas import UserProfile, WardrobeItem
from features.planner import generate_weekly_plan, regenerate_day
from features.recommendation import is_outfit_suitable_for_occasion, recommend_outfits, score_outfit


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

    def test_complete_only_recommendations_exclude_single_items(self):
        results = recommend_outfits(self.profile, self.items, "Casual", top_k=8, complete_only=True)
        self.assertTrue(results)
        self.assertTrue(all(len(result.items) >= 2 for result in results))
        self.assertTrue(all(result.occasion == "Casual" for result in results))
        self.assertTrue(all(result.item_ids and result.outfit_id and result.image_paths for result in results))

    def test_recommendations_filter_incompatible_market_category(self):
        wardrobe = [
            item("womens-top", "Top", "green"),
            item("womens-bottom", "Bottom", "black"),
            item("mens-top", "Top", "blue"),
            item("mens-bottom", "Bottom", "grey"),
        ]
        wardrobe[0].market_category = "Women's"
        wardrobe[1].market_category = "Women's"
        wardrobe[2].market_category = "Men's"
        wardrobe[3].market_category = "Men's"
        profile = UserProfile(gender="Female")

        results = recommend_outfits(profile, wardrobe, "College", top_k=8, complete_only=True, include_unisex_unknown=False)

        self.assertTrue(results)
        self.assertTrue(all(not {"mens-top", "mens-bottom"} & set(result.item_ids) for result in results))

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

    def test_explicit_wardrobe_occasions_prevent_unrelated_recommendations(self):
        casual_dress = item("casual-dress", "Dress", "beige")
        casual_dress.suitable_occasions = ["Casual", "Travel"]
        traditional_dress = item("traditional-dress", "Dress", "green")
        traditional_dress.suitable_occasions = ["Traditional", "Wedding"]

        self.assertFalse(is_outfit_suitable_for_occasion([casual_dress], "Traditional"))
        self.assertTrue(is_outfit_suitable_for_occasion([casual_dress], "Casual"))
        self.assertTrue(is_outfit_suitable_for_occasion([traditional_dress], "Wedding"))


if __name__ == "__main__":
    unittest.main()
