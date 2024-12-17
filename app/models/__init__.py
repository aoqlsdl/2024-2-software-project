from app import db
from app.models.disabled_course import DisabledCourse
from app.models.class_time import ClassTime
from app.models.course_class import course_class

# 관계 설정
DisabledCourse.class_times = db.relationship(
    'ClassTime', secondary=course_class, back_populates='disabled_courses'
)
ClassTime.disabled_courses = db.relationship(
    'DisabledCourse', secondary=course_class, back_populates='class_times'
)