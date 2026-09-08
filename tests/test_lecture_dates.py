import unittest
from src.icourse import ICourseClient


class Response:
    def raise_for_status(self):
        pass

    def json(self):
        return {"code": 0, "data": {"title": "测试课程", "realname": "教师甲", "sub_list": {
            "2026": {"9": {"2": [
                {"id": "1", "sub_title": "2026-09-07第3-5节", "playback_status": 1},
                {"id": "2", "sub_title": "尚无日期", "playback_status": 0}]}}}}}


class Session:
    def get(self, *args, **kwargs):
        return Response()


class LectureDateTests(unittest.TestCase):
    def test_week_group_is_not_used_as_calendar_day(self):
        lectures = ICourseClient(Session()).get_course_detail("123")["lectures"]
        self.assertEqual(lectures[0]["date"], "2026-09-07")
        self.assertTrue(lectures[0]["has_playback"])
        self.assertEqual(lectures[1]["date"], "")
        self.assertFalse(lectures[1]["has_playback"])
