# -*- coding: utf-8 -*-
from string import letters, digits, whitespace
import sys

table = {}  #전역 변수를 저장할 dictionary
local_table={}  #함수의 지역변수를 구현하기위한 dictionary

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

        def build_token(type, lexeme): return Token(type, lexeme)
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

    def hasNext(self):
        if(self.next is None):
            return False
        else:
            return True

    def __str__(self):
        result = ''

        if self.type is TokenType.ID:
            result = '[' + NODETYPE_NAMES[self.type] + ':' + self.value + ']'
        elif self.type is TokenType.INT:
            result = '['+NODETYPE_NAMES[self.type]+':' + self.value + ']'
        elif self.type is TokenType.LIST:
            if self.value is not None:
                if self.value.type is TokenType.QUOTE:
                    result = str(self.value)
                else:
                    result = '(' + str(self.value) + ')'
            else:
                result = '(' + str(self.value) + ')'
        elif self.type is TokenType.QUOTE:
            result = "\'"
        else:
            result = '['+NODETYPE_NAMES[self.type]+']'

        if self.next is not None:
            return result + ' ' + str(self.next)
        else:
            return result

    def lookupTable(self,value):
        return table[value]


class BasicPaser(object):

    def __init__(self, token_list):
        self.token_iter = iter(token_list)

    def _get_next_token(self):
        next_token = next(self.token_iter, None)
        if next_token is None:
            return None
        return next_token

    def parse_expr(self):
        token = self._get_next_token()

        if token is None:
            return None
        result = self._create_node(token)
        return result

    def _create_node(self, token):
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
        elif token.type in CuteType.BOOLEAN_LIST:
            return Node(token.type)
        elif token.type in CuteType.BINARYOP_LIST:
            return Node(token.type, token.lexeme)
        elif token.type is CuteType.QUOTE:
            return Node(TokenType.QUOTE, token.lexeme)
        elif token.type is CuteType.APOSTROPHE:
            node = Node(TokenType.LIST, Node(TokenType.QUOTE, token.lexeme))
            node.value.next = self.parse_expr()
            return node
        elif check_keyword(token.lexeme):
            return Node(token.type, token.lexeme)

    def _parse_expr_list(self):
        head = self.parse_expr()
        '"":type :Node""'
        if head is not None:
            head.next = self._parse_expr_list()
        return head




def run_list(root_node):
    op_code_node = root_node.value
    return run_func(op_code_node)(root_node)


def run_func(op_code_node):
    def quote(node):
        return node

    def strip_quote(node):
        if node.type is TokenType.LIST:
            if node.value is TokenType.QUOTE or TokenType.APOSTROPHE:
                return node.value.next
        if node.type is TokenType.QUOTE:
            return node.next
        return node

    def cons(node):
        l_node = node.value.next
        r_node = l_node.next
        r_node = run_expr(r_node)
        l_node = run_expr(l_node)
        new_r_node = r_node
        new_l_node = l_node
        new_r_node = strip_quote(new_r_node)
        new_l_node = strip_quote(new_l_node)
        new_l_node.next = new_r_node.value

        return create_new_quote_list(new_l_node, True)

    def car(node):
        l_node = run_expr(node.value.next)
        result = strip_quote(l_node).value

        if result.type is not TokenType.LIST:
            return result
        return create_new_quote_list(result)

    def cdr(node):
        l_node = node.value.next
        l_node = run_expr(l_node)
        new_r_node = strip_quote(l_node)
        return create_new_quote_list(new_r_node.value.next, True)

    def null_q(node):
        l_node = run_expr(node.value.next)
        new_l_node = strip_quote(l_node).value
        if new_l_node is None:
            return Node(TokenType.TRUE)
        else:
            return Node(TokenType.FALSE)

    def atom_q(node):
        l_node = run_expr(node.value.next)
        new_l_node = strip_quote(l_node)

        if new_l_node.type is TokenType.LIST:
            if new_l_node.value is None:
                return Node(TokenType.TRUE)
            return Node(TokenType.FALSE)
        else:
            return Node(TokenType.TRUE)

    def eq_q(node):
        l_node = node.value.next
        r_node = l_node.next
        new_l_node = strip_quote(run_expr(l_node))
        new_r_node = strip_quote(run_expr(r_node))

        if (new_l_node.type or new_r_node.type) is not TokenType.INT:
            return Node(TokenType.FALSE)
        if new_l_node.value == new_r_node.value:
            return Node(TokenType.TRUE)
        return Node(TokenType.FALSE)

    def plus(node):
        l_node = run_expr(node.value.next)
        r_node = run_expr(node.value.next.next)
        return Node(TokenType.INT, int(l_node.value)+int(r_node.value))

    def minus(node):
        l_node = run_expr(node.value.next)
        r_node = run_expr(node.value.next.next)
        return Node(TokenType.INT, int(l_node.value) - int(r_node.value))

    def multiple(node):
        l_node = run_expr(node.value.next)
        r_node = run_expr(node.value.next.next)
        return Node(TokenType.INT, int(l_node.value) * int(r_node.value))

    def divide(node):
        l_node = run_expr(node.value.next)
        r_node = run_expr(node.value.next.next)
        return Node(TokenType.INT, int(l_node.value) / int(r_node.value))

    def not_op(node):
        l_node = run_expr(node.value.next)
        if l_node.type is TokenType.FALSE:
            return Node(TokenType.TRUE)
        return Node(TokenType.FALSE)

    def lt(node):
        l_node = run_expr(node.value.next)
        r_node = run_expr(node.value.next.next)
        if int(l_node.value) < int(r_node.value):
            return Node(TokenType.TRUE)
        return Node(TokenType.FALSE)

    def gt(node):
        l_node = run_expr(node.value.next)
        r_node = run_expr(node.value.next.next)
        if int(l_node.value) > int(r_node.value):
            return Node(TokenType.TRUE)
        return Node(TokenType.FALSE)

    def eq(node):
        l_node = run_expr(node.value.next.next)
        r_node = run_expr(node.value.next)
        if int(l_node.value) == int(r_node.value):
            return Node(TokenType.TRUE)
        return Node(TokenType.FALSE)

    def cond(node):
        l_node = node.value.next
        if l_node is not None:
            return run_cond(l_node)
        else:
            print('cond null error!')

    def run_cond(node):
        l_node = run_expr(node.value)
        if l_node.type is TokenType.TRUE:
            return run_expr(node.value.next)
        if node.next is None:
            print "Error"
            return None
        return run_cond(node.next)

    def define(node):
        key = node.value.next
        value = node.value.next.next
        insertTable(key.value,value)

    def insertTable(id, value):
        table[id] = value


    def local_define(node):
        key = node.value.next
        value = node.value.next.next
        insert_localTable(key.value, value)

    def insert_localTable(id,value):
        local_table[id] =value

    def runmethod(node):    #람다일때 실행

        if local_table.has_key(node.value.value):
            parameter = local_table[node.value.value].value.next.value  # 파라미터 변수
            run = local_table[node.value.value].value.next.next  # 실행문
        else:
            parameter = table[node.value.value].value.next.value  # 파라미터 변수
            run = table[node.value.value].value.next.next  # 실행문
        if run.value.type is TokenType.DEFINE:  #만약 실행문이 DEFINE일경우

            key = run.value.next
            value = run.value.next.next
            insert_localTable(key.value,value)
            run=run.next

        if parameter.hasNext():
            if parameter.next.hasNext():
                parameter = parameter.next.next
                insert_localTable(parameter.value, run_expr(node.value.next.next.next))
                fun1 = node.value.next
                fun2 = node.value.next.next
                run.value.value = fun2.value
                run.value.next.value.value = fun1.value
            else:
                insert_localTable(parameter.value, run_expr(node.value.next))
                func = node.value.next.next
                run.value.value = func.value
        else:
            insert_localTable(parameter.value, run_expr(node.value.next))
        result = run_expr(run)
        return result


    def run_lambda(node):
        if node.value.type is TokenType.LAMBDA:
            return node
        elif node.value.value.type is TokenType.LAMBDA:
            parameter = node.value.value.next.value  # 파라미터 변수
            binding = node.value.next  # 두번째 파라미터 값
            run = node.value.value.next.next  # 실행문
            if binding is None:
                return node
            while parameter is not None:
                insert_localTable(parameter.value, binding)
                parameter = parameter.next  # 두번째 파라미터 변수
                binding = binding.next
            return run_expr(run)


    def create_new_quote_list(value_node, list_flag=False):
        quote_list = Node(TokenType.QUOTE, 'quote')
        wrapper_new_list = Node(TokenType.LIST, quote_list)
        if value_node is None:
            pass
        elif value_node.type is TokenType.LIST:
            if list_flag:
                inner_l_node = Node(TokenType.LIST, value_node)
                quote_list.next = inner_l_node
            else:
                quote_list.next = value_node
            return wrapper_new_list
        new_value_list = Node(TokenType.LIST, value_node)
        quote_list.next = new_value_list
        return wrapper_new_list

    table['cons'] = cons
    table["'"] = quote
    table['quote'] = quote
    table['cdr'] = cdr
    table['car'] = car
    table['eq?'] = eq_q
    table['null?'] = null_q
    table['atom?'] = atom_q
    table['not'] = not_op
    table['+'] = plus
    table['-'] = minus
    table['*'] = multiple
    table['/'] = divide
    table['<'] = lt
    table['>'] = gt
    table['='] = eq
    table['cond'] = cond
    table['define'] = define


    if local_table.has_key(op_code_node.value):
        if type(local_table[op_code_node.value]) is Node:
            if local_table[op_code_node.value].value.type is TokenType.LAMBDA:
                return runmethod
        return local_table[op_code_node.value]

    elif table.has_key(op_code_node.value):
        if type(table[op_code_node.value]) is Node:
            if table[op_code_node.value].value.type is TokenType.LAMBDA:
                return runmethod
        return table[op_code_node.value]

    elif op_code_node.type is TokenType.LAMBDA:
        return run_lambda
    elif op_code_node.value.type is TokenType.LAMBDA:
        return  run_lambda

def lookuptable(str):
    if table.has_key(str):
        return True
    else:
        return False

def lookuptable_local(str):
    if local_table.has_key(str):
        return True
    else:
        return False

def run_expr(root_node):
    """
    :type root_node : Node
    """
    if root_node is None:
        return None

    if root_node.type is TokenType.ID:
        if lookuptable_local(root_node.value):
            return run_expr(local_table[root_node.value])
        if lookuptable(root_node.value):
            return run_expr(table[root_node.value])
        return root_node
    elif root_node.type is TokenType.INT:
        return root_node
    elif root_node.type is TokenType.TRUE:
        return root_node
    elif root_node.type is TokenType.FALSE:
        return root_node
    elif root_node.type is TokenType.LIST:
        return run_list(root_node)

    else:
        print 'Run Expr Error'
    return None


def print_node(node):

    def print_list(node):
        def print_list_val(node):
            if node.next is not None:
                return print_node(node)+' '+print_list_val(node.next)
            return print_node(node)

        if node.type is TokenType.LIST:
            if node.value is None:
                return '( )'
            if node.value.type is TokenType.QUOTE:
                return print_node(node.value)
            return '('+print_list_val(node.value)+')'

    if node is None:
        return ''
    if node.type in [TokenType.ID, TokenType.INT]:
        return node.value
    if node.type is TokenType.TRUE:
        return '#T'
    if node.type is TokenType.FALSE:
        return '#F'
    if node.type is TokenType.PLUS:
        return '+'
    if node.type is TokenType.MINUS:
        return '-'
    if node.type is TokenType.TIMES:
        return '*'
    if node.type is TokenType.DIV:
        return '/'
    if node.type is TokenType.GT:
        return '>'
    if node.type is TokenType.LT:
        return '<'
    if node.type is TokenType.EQ:
        return '='
    if node.type is TokenType.LIST:
        return print_list(node)
    if node.type is TokenType.ATOM_Q:
        return 'atom?'
    if node.type is TokenType.CAR:
        return 'car'
    if node.type is TokenType.CDR:
        return 'cdr'
    if node.type is TokenType.COND:
        return 'cond'
    if node.type is TokenType.CONS:
        return 'cons'
    if node.type is TokenType.LAMBDA:
        return 'lambda'
    if node.type is TokenType.NULL_Q:
        return 'null?'
    if node.type is TokenType.EQ_Q:
        return 'eq?'
    if node.type is TokenType.NOT:
        return 'not'
    if node.type is TokenType.QUOTE:
        return "'"+print_node(node.next)

def remove_local():
    local_table.clear()

def Test_method(input):
    test_cute = CuteScanner(input)
    test_tokens = test_cute.tokenize()
    test_basic_paser = BasicPaser(test_tokens)
    node = test_basic_paser.parse_expr()
    cute_inter = run_expr(node)
    print print_node(cute_inter)
    remove_local()



def Test_All():

    while True :
        input_commend = raw_input("> ")
        if input_commend == 'exit':
            break;
        import sys
        sys.stdout.write("... ")
        Test_method(input_commend)

    # Test_method("( define a 1)")
    # Test_method((" a "))
    # Test_method("( define b '(1 2 3)")
    # Test_method(" b ")
    # Test_method("( define c ( - 5 2 )")
    # Test_method("c")
    # Test_method("( define d '(+ 2 3) )")
    # Test_method("d")
    # Test_method("( define test b )")
    # Test_method("test")
    # Test_method("( define a 2)")
    # Test_method("( * a 4 )")
    # Test_method("( ( lambda ( x ) (* x -2) ) 3)")
    # Test_method("( ( lambda (x) (/ x 2)) a )")
    # Test_method("( ( lambda (x y) (* x y)) 3 5)")
    # Test_method("( ( lambda (x y) (* x y)) a 5)")
    # Test_method("( define plus1 ( lambda ( x ) ( + x 1 ) ) ) ")
    # Test_method("( plus1 3 )")
    # Test_method("(define mul1 (lambda (x) (* x a))) ")
    # Test_method("( mul1 a )")
    # Test_method("( define plus2 ( lambda ( x ) (+ ( plus1 x ) 1")
    # Test_method("( plus2 4 )")
    # Test_method("(define plus3 ( lambda (x) (+ (plus1 x) a)))")
    # Test_method("( plus3 a )")
    # Test_method("(define mul2 ( lambda (x) (* (plus1 x) -2)))")
    # Test_method("( mul2 7 )")
    # Test_method("( define lastitem ( lambda (ls) (cond ( ( null? ( cdr ls ) ) ( car ls ) ) ( #T ( lastitem (cdr ls ) ) ) ) ) )")
    # Test_method("( lastitem b )")
    # Test_method("( define square ( lambda (x) (* x x)))")
    # Test_method("( define yourfunc ( lambda (x func) (func x))")
    # Test_method("( yourfunc 3 square )")
    # Test_method("( define multwo ( lambda (x) ( * 2 x ) ) )")
    # Test_method("( define newfun ( lambda ( fun1 fun2 x ) ( fun2 (fun1 x ) ) ) )")
    # Test_method("( newfun square multwo 10)")
    # Test_method("( define cube ( lambda ( n ) ( define sqrt ( lambda ( n ) ( * n n) ) ) ( * ( sqrt n ) n ) ) )")
    # Test_method("( cube 4 )")
    # Test_method("( sqrt 4 )")


Test_All()