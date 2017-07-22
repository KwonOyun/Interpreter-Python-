
class TokenType():
    ID=3
    INT=2

TOKENTYPE_NAMES={ 2: "INT", 3:"ID"}

class Token():

    def __init__(self, type, lexeme):
        self.type= type
        self.lexeme= lexeme

    def __str__(self):
        # return self.lexeme
        return "[" + TOKENTYPE_NAMES[self.type] + ": " + self.lexeme + "]"

    def __repr__(self):
        return self.__str__()


class CuteScanner():

    def __init__(self, source):
        source = source.strip()
        # tokenize a string, delimiter is " "
        token_list = source.split(" ")
        # iterator
        self.token_iter = iter (token_list)

    def next_token(self):
        state=0
        # get if token exist
        temp_token = next(self.token_iter, None)

        if temp_token is None :
            return None
        for temp_char in temp_token:
            """:type : str"""
            if state==0:
                if temp_char.isdigit(): state=2
                elif temp_char=='-': state=1
                elif temp_char.isalpha():state=3
                else :
                    print "ERROR"
                    return None
            elif state==1:
                if temp_char.isdigit(): state=2
                    # fill out if state is 1
                else :
                    print "ERROR"
                    return None
            elif state==2: state=2
            # fill out if state is 2

            elif state==3: state=3
            # fill out if state is 3

            else:
                print "ERROR"
                return None

        if state ==2:
            return Token(TokenType.INT, temp_token)
        elif state==3:
            return Token(TokenType.ID, temp_token)

    def tokenize(self):
        #Type is List
        tokens=[]
        while True :
            # fill out
            temp = self.next_token()
            if temp is None:
                break
            tokens.append(temp)
        return tokens

def Test_CuteScanner():
    test_cute = CuteScanner("banana 267 h cat -3789 7 y2010")
    test_tokens=test_cute. tokenize()

    print test_tokens
    for token_i in test_tokens:
        print token_i
    print "end"

Test_CuteScanner()