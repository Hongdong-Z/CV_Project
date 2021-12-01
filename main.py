import face_recognition


def distance(first_dir,second_dir):
    known_image = face_recognition.load_image_file(first_dir)
    unknown_image = face_recognition.load_image_file(second_dir)
    default_value = face_recognition.load_image_file('0eSUzIMKjIrt.jpg')
    default_encoding = face_recognition.face_encodings(default_value)
    first_encoding = face_recognition.face_encodings(known_image)
    second_encoding = face_recognition.face_encodings(unknown_image)
    if len(first_encoding) > 0:
        first_e = first_encoding[0]
    else:
        print(first_dir)
        first_e = default_encoding[0]
    if len(second_encoding) > 0:
        second_e = second_encoding[0]
    else:
        print(second_dir)
        second_e = default_encoding[0]
    distance = face_recognition.face_distance([first_e],second_e )
    return distance


filename = 'testPairs'
result = 'result.txt'
with open(result,'w') as result:
    with open(filename) as pairs:
        for line in pairs:
            split_name = line.split( )
            first_name = split_name[0]
            second_name = split_name[1]
            first_name_dir = 'data/test'+"/"+first_name
            second_name_dir = 'data/test'+'/'+second_name
            non_similarity = distance(first_name_dir,second_name_dir)
            similarity = 1 - non_similarity[0]
            #print(similarity)
            result.write(str(similarity)+'\n')