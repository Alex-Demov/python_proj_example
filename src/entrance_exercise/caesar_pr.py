def caesar(shift, text):
    alphabet=' abcdefghijklmnopqrstuvwxyz'
    decrypted_text=''

    for char in text:
        decrypted_text += alphabet[(alphabet.index(char)+shift) % len(alphabet)]
    return decrypted_text

shift = int(input())
text = input().strip()

result = caesar(shift, text)
print(result)