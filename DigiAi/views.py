from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser
from django.http.response import JsonResponse
from DigiAi.serializers import StudentSerializer
from DigiAi.models import Student

from django.http import Http404

@csrf_exempt
def studentApi(request,id=0):
    if request.method=='GET':
        students = Student.objects.all()
        student_serializer=StudentSerializer(students,many=True)
        return JsonResponse(student_serializer.data,safe=False)
    elif request.method=='POST':
        student_data=JSONParser().parse(request)
        student_serializer=StudentSerializer(data=student_data)
        if student_serializer.is_valid():
            student_serializer.save()
            return JsonResponse("Added Successfully",safe=False)
        return JsonResponse("Failed to Add",safe=False)
    elif request.method=='PUT':
        try:
            student=Student.objects.get(id=id)
            student_data=JSONParser().parse(request)
            student_serializer=StudentSerializer(student,data=student_data)
            if student_serializer.is_valid():
                student_serializer.save()
                return JsonResponse("Updated Successfully",safe=False)
        except Student.DoesNotExist:
            raise Http404("Student does not exist")
        return JsonResponse("Failed to Update")
    elif request.method=='DELETE':
        try:
            student=Student.objects.get(id=id)
            student.delete()
            return JsonResponse("Deleted Successfully",safe=False)
        except Student.DoesNotExist:
            raise Http404("Student does not exist")
