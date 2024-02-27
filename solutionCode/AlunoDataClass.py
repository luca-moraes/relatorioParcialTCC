from dataclasses import dataclass
from typing import List

@dataclass
class Answer:
    number_question : int
    answer_question : str

@dataclass
class Student:
    identification_student : int
    responses_student : List[Answer]

def list_students_responser(students_list: List[Student]) -> None:
    for student in students_list:
        print(str(student.identification_student) + " responses: \n")

        for answer in student.responses_student:
            print("Question " + str(answer.number_question) + " response: ", end="")
            print(answer.answer_question + "\n")
    pass


response1 = Answer(1,"r1")
response2 = Answer(2,"r2")
student1 = Student(1, [response1, response2])

response3 = Answer(1,"r3")
response4 = Answer(2,"r4")
student2 = Student(2, [response3, response4])

response5 = Answer(1,"r5")
response6 = Answer(2,"r6")
student3 = Student(3, [response5, response6])

list_students_responser([student1, student2, student3])