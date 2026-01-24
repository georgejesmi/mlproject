import sys
from src.logger import logger

def error_msg_details(error, error_details:sys):
    _, _, tb = error_details.exc_info()
    file_name = tb.tb_frame.f_code.co_filename
    error_msg = f"{file_name} : {tb.tb_frame.f_lineno} : {error}"
    return error_msg

class CustomException(Exception):
    def __init__(self, error, error_details:sys):
        super().__init__(error)
        self.error_msg = error_msg_details(error, error_details)

    def __str__(self):
        return self.error_msg

# Sample exception in code
# try:
#     a = 1 / 0  # Intentional division by zero error
# except Exception as e:
#     raise CustomException(e, sys)

"""
raising a custom exception
when we try to print the object a class
eg : class Student:
        def __init__(self, name, age):
            self.name = name
            self.age = age
    student1 = Student("John", 25)
    print(student1)
The above code prints <__main__.ClassName object at 0x...>
ie, it displays object's type and its memory address
But if we use, __str__() , the object details can be printed in user friendly manner
eg : class Student():
        def __init__(self, name, age):
            self.name = name
            self.age = age
        def __str__(self):
            return f"{self.name} : {self.age}"
     student1 = Student("John", 25)
     print(student1)
Output = John : 25
"""