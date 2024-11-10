from rest_framework import serializers


class FileSerializer(serializers.Serializer):
    file = serializers.ImageField(max_length=None, use_url=True)
    
    def create(self, validated_data):
        print("Validated data: ", validated_data) 
        file = validated_data.get('file', None)
        if file is not None:
            print("File name: ", file.name)  
        else:
            print("File is None")  
            return {'file': None}
