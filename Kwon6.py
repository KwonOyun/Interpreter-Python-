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

    BINARYOP_LIST = (DIV, TIMES, MINUS, PLUS, LT, GT, EQ)
    BOOLEAN_LIST = (TRUE, FALSE)


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
    CuteType()) if not callable(attr) and not attr.startswith('__'))


class Token(object):
    def __init__(self, type, lexeme):
        """
        :type type:CuteType
        :type lexeme: str
        :param type:
        :param lexeme:
        :return:
        """
        if check_keyword(lexeme):
            self.type = _get_keyword_type(lexeme)
            self.lexeme = lexeme
        else:
            self.type = type
            self.lexeme = lexeme
        # print type
        print '[' + CUTETYPE_NAMES[self.type] + ': ' + self.lexeme + ']'

    def __str__(self):
        # return self.lexeme
        return '[' + CUTETYPE_NAMES[self.type] + ': ' + self.lexeme + ']'

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
        temp_char = ''
        return_token = ''
        while not self.eos():
            temp_char = self.get()
            if old_state == 0 and temp_char in (')', '('):
                return_token = temp_char
                old_state = transition_matrix[(old_state, temp_char)]
                break

            return_token += temp_char
            old_state = transition_matrix[(old_state, temp_char)]
            next_char = self.peek()
            if next_char in whitespace or next_char in ('(', ')'):
                break

        return build_token_func(old_state, return_token)

    def scan(self, transition_matrix, build_token_func):
        print 'scanning...'
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

        self.transM[(4, '?')] = 16
        self.transM[(0, '-')] = 2
        self.transM[(0, '+')] = 3
        self.transM[(0, '(')] = 5
        self.transM[(0, ')')] = 6

        self.transM[(0, '#')] = 7
        self.transM[(7, 'T')] = 8
        self.transM[(7, 'F')] = 9

        self.transM[(0, '/')] = 11
        self.transM[(0, '*')] = 10

        self.transM[(0, '<')] = 12
        self.transM[(0, '>')] = 13
        self.transM[(0, '=')] = 14
        self.transM[(0, "'")] = 15

    def tokenize(self):
        print '  ===  tokenize  === '

        def build_token(type, lexeme): return Token(type, lexeme)
        print self.source
        cute_scanner = Scanner(self.source)
        return cute_scanner.scan(self.transM, build_token)


class TokenType():
    INT = 1
    ID = 4
    MINUS = 2
    PLUS = 3
    LIST = 5
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


NODETYPE_NAMES = dict((eval(attr, globals(), TokenType.__dict__), attr) for attr in dir(
    TokenType()) if not callable(attr) and not attr.startswith('__'))


class Node (object):

    def __init__(self, type, value=None):
        self.next = None
        self.value = value
        self.type = type

    def set_last_next(self, next_node):
        if self.next is not None:
            self.next.set_last_next(next_node)

        else:
            self.next = next_node

    def __str__(self):
        result = ''

        if self.type is TokenType.ID:
            result = '[' + NODETYPE_NAMES[self.type] + ':' + self.value + ']'
        elif self.type is TokenType.INT:
            result = '[' + NODETYPE_NAMES[self.type] + ':' + self.value + ']'
        elif self.type is TokenType.LIST:
            if self.value.type is TokenType.QUOTE:
                result = str(self.value)
            else:
                result = '('+str(self.value)+')'
        elif self.type is TokenType.QUOTE:
            result = "'" + str(self.value)

        else:
            result = "["+NODETYPE_NAMES[self.type]+"]"

        # fill out
        # next 노드에 대해서도 출력하도록 작성
        # recursion 이용
        #Quote에 대해서도 출력하도록 작성

        if self.next is not None:
            result = result + '' + str(self.next)
        return result


class BasicPaser(object):

    def __init__(self, token_list):
        """
        :type token_list:list
        :param token_list:
        :return:
        """
        self.token_iter = iter(token_list)

    def _get_next_token(self):
        """
        :rtype: Token
        :return:
        """
        next_token = next(self.token_iter, None)
        if next_token is None:
            return None
        return next_token

    def parse_expr(self):
        """
        :rtype : Node
        :return:
        """
        token = self._get_next_token()
        # 하나의 Token을 create_node()를 이용하여 Node로 만들어 반환함
        '"":type :Token""'
        if token is None:
            return None
        result = self._create_node(token)
        return result

    def _create_node(self, token):
        # 토큰을 Node로 만듬
        if token is None:
            return None
        elif token.type is CuteType.INT:
            return Node(TokenType.INT,  token.lexeme)
        elif token.type is CuteType.ID:
            return Node(TokenType.ID,   token.lexeme)
        elif token.type is CuteType.L_PAREN:
            return Node(TokenType.LIST, self._parse_expr_list())
        elif token.type is CuteType.R_PAREN:
            return None
        elif  token.type is CuteType.TRUE:
            return Node(TokenType.TRUE, token.lexeme)
        elif token.type is CuteType.FALSE:
            return Node(TokenType.FALSE, token.lexeme)
        elif token.type is CuteType.TIMES:
            return Node(TokenType.TIMES, token.lexeme)
        elif token.type is CuteType.DIV:
            return Node(TokenType.DIV, token.lexeme)
        elif token.type is CuteType.LT:
            return Node(TokenType.LT, token.lexeme)
        elif token.type is CuteType.GT:
            return Node(TokenType.GT, token.lexeme)
        elif token.type is CuteType.EQ:
            return Node(TokenType.EQ, token.lexeme)
        elif token.type is CuteType.DEFINE:
            return Node(TokenType.DEFINE, token.lexeme)
        elif token.type is CuteType.LAMBDA:
            return Node(TokenType.LAMBDA, token.lexeme)
        elif token.type is CuteType.COND:
            return Node(TokenType.COND, token.lexeme)
        elif token.type is CuteType.NOT:
            return Node(TokenType.NOT, token.lexeme)
        elif token.type is CuteType.CAR:
            return Node(TokenType.CAR, token.lexeme)
        elif token.type is CuteType.CDR:
            return Node(TokenType.CDR, token.lexeme)
        elif token.type is CuteType.CONS:
            return Node(TokenType.CONS, token.lexeme)
        elif token.type is CuteType.ATOM_Q:
            return Node(TokenType.ATOM_Q, token.lexeme)
        elif token.type is CuteType.NULL_Q:
            return Node(TokenType.NULL_Q, token.lexeme)
        elif token.type is CuteType.EQ_Q:
            return Node(TokenType.EQ_Q, token.lexeme)
        elif token.type is CuteType.PLUS:
            return Node(TokenType.PLUS, token.lexeme)
        elif token.type is CuteType.QUOTE:
            return Node(TokenType.QUOTE, self._parse_expr_list())
        elif token.type is CuteType.APOSTROPHE:
            return Node(TokenType.QUOTE, self._parse_expr_list())

            # 조건 작성, INT, ID, L_PAREN, R_PAREN을 제외한 나머지 경우:

            # “#T”, “#F”, ‘+’, ‘-‘, ‘*’, ‘/’, ‘<’, ‘>’, ‘=’, “define”, “cond”, “not”, “car”,
            # “cdr”, “cons”, “eq?”, “atom?”, “null?”, 에 대해서 작성
            # 조건에 맞는 노드를 반환하도록 작성
            # 기호 ‘와 quote에 대해서도 작성
            # quote의 노드 구조에 맞게 작성(Quote 내부구조 참조)

        else:
            return None

    def _parse_expr_list(self):
        # Token이 ‘(‘일 경우, list 형태로 만들어서 반환
        head = self.parse_expr()
        '"":type :Node""'
        if head is not None:
            head.next = self._parse_expr_list()
        return head


def Test_BasicPaser():
    test_cute = CuteScanner(" ' ( + 3 2 ) ")
    test_tokens = test_cute.tokenize()
    print test_tokens
    test_basic_paser = BasicPaser(test_tokens)
    node = test_basic_paser.parse_expr()
    print node


Test_BasicPaser()
