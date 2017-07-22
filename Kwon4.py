# -*- coding: utf-8 -*-
from string import letters, digits, whitespace


class CuteType:
    INT = 1
    ID = 4

    MINUS = 2
    PLUS = 3

    L_PAREN = 5
    R_PAREN = 6

    TRUE = 8
    FALSE = 9

    TIMES = 10
    DIV = 11

    LT = 12
    GT = 13
    EQ = 14
    APOSTROPHE = 15

    DEFINE = 20
    LAMBDA = 21
    COND = 22
    QUOTE = 23
    NOT = 24
    CAR = 25
    CDR = 26
    CONS = 27
    ATOM_Q = 28
    NULL_Q = 29
    EQ_Q = 30

    KEYWORD_LIST = ('define', 'lambda', 'cond', 'quote', 'not', 'car', 'cdr', 'cons',
                    'atom?', 'null?', 'eq?')


def check_keyword(token):
    """
    :type token:str
    :param token:
    :return:
    """
    if token.lower() in CuteType.KEYWORD_LIST:
        return True
    return False


def _get_keyword_type(token):
    return {
        'define': CuteType.DEFINE,
        'lambda': CuteType.LAMBDA,
        'cond': CuteType.COND,
        'quote': CuteType.QUOTE,
        'not': CuteType.NOT,
        'car': CuteType.CAR,
        'cdr': CuteType.CDR,
        'cons': CuteType.CONS,
        'atom?': CuteType.ATOM_Q,
        'null?': CuteType.NULL_Q,
        'eq?': CuteType.EQ_Q
    }[token]

CUTETYPE_NAMES = dict((eval(attr, globals(), CuteType.__dict__), attr) for attr in dir(
    CuteType()) if not callable(attr) and not attr.startswith("__"))


class Token(object):
    def __init__(self, type, lexeme):
        """
        :type type:CuteType
        :type lexeme: str
        :param type:
        :param lexeme:
        :return:
        """
        # Initialize the token to generate.
		# Fill Out

        if check_keyword(lexeme):
            type = _get_keyword_type(lexeme)
        self.type = type
        self.lexeme = lexeme

        print "[" + CUTETYPE_NAMES[self.type] + ": " + self.lexeme + "]"

    def __str__(self):
        # return self.lexeme
        return "[" + CUTETYPE_NAMES[self.type] + ": " + self.lexeme + "]"

    def __repr__(self):
        return str(self)


class Scanner:

    def __init__(self, source_string=None):
        """
        :type self.__source_string: str
        :param source_string:
        """
        self.__source_string = source_string
        self.__pos = 0
        self.__length = len(source_string)
        self.__token_list = []

    def __make_token(self, transition_matrix, build_token_func=None):
        old_state = 0
        self.__skip_whitespace()
        temp_char = ""
        return_token = ""
        while not self.eos():
            temp_char = self.get()
            if old_state == 0 and temp_char in (")", "("):
                return_token = temp_char
                old_state = transition_matrix[(old_state, temp_char)]
                break

            return_token += temp_char
            old_state = transition_matrix[(old_state, temp_char)]
            next_char = self.peek()
            if next_char in whitespace or next_char in ("(", ")"):
                break

        return build_token_func(old_state, return_token)

    def scan(self, transition_matrix, build_token_func):
        print "scanning..."
        while not self.eos():
            self.__token_list.append(self.__make_token(
                transition_matrix, build_token_func))
        return self.__token_list

    def pos(self):
        return self.__pos

    def eos(self):
        return self.__pos >= self.__length

    def skip(self, pattern):
        while not self.eos():
            temp_char = self.peek()
            if temp_char in pattern:
                temp_char = self.get()
            else:
                break

    def __skip_whitespace(self):
        self.skip(whitespace)

    def peek(self, length=1):
        return self.__source_string[self.__pos: self.__pos + length]

    def get(self, length=1):
        return_get_string = self.peek(length)
        self.__pos += len(return_get_string)
        return return_get_string


class CuteScanner(object):

    transM = {}

    def __init__(self, source):
        """
        :type source:str
        :param source:
        :return:
        """
        self.source = source
        self._init_TM()

    def _init_TM(self):
        for alpha in letters:
            self.transM[(0, alpha)] = 4
            self.transM[(4, alpha)] = 4

        for digit in digits:
            self.transM[(0, digit)] = 1
            self.transM[(1, digit)] = 1
            self.transM[(2, digit)] = 1
            self.transM[(4, digit)] = 4


        # Complete the remaining transition matrix
		# Fill Out
        self.transM[(0,')')] = 6
        self.transM[(0, '#')] = 7
        self.transM[(0, '(')] = 5
        self.transM[(0, '+')] = 3
        self.transM[(0, '-')] = 2
        self.transM[(4, '?')] = 16
        self.transM[(0, '<')] = 12
        self.transM[(0, '=')] = 14
        self.transM[(0, '>')] = 13
        self.transM[(0, "'")] = 15
        self.transM[(0, '/')] = 11
        self.transM[(0, '*')] = 10
        self.transM[(7, 'T')] = 8
        self.transM[(7, 'F')] = 9


    def tokenize(self):
        print "  ===  tokenize  === "

        def build_token(type,lexeme): return Token(type,lexeme)
        print self.source
        cute_scanner = Scanner(self.source)
        return cute_scanner.scan(self.transM, build_token)


test_cute = CuteScanner("Test car + ' - * #T ( ) eq?")
test_tokens = test_cute.tokenize()
print test_tokens
