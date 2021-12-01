import face_recognition
def distance(first_dir,second_dir):
    known_image = face_recognition.load_image_file(first_dir)
    unknown_image = face_recognition.load_image_file(second_dir)

    biden_encoding = face_recognition.face_encodings(known_image)[0]
    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

    distance = face_recognition.face_distance([biden_encoding], unknown_encoding)
    return distance
i = distance('0eSUzIMKjIrt.jpg','Adel_Al-Jubeir_0002.jpg')
print(i[0])
print(1-i[0])

