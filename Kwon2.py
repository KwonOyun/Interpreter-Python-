
import random

class RecursionLinkedList(object):

    def __init__(self):
        self.head = None

    def _link_first(self, element):
        # connect newly created node the beginning of the list
        if self is not None:
            self.head = Node(element, self.head)
        else:
            self.head = Node(element, None)

    def _link_last(self, element, node):
        """
        :type node: Node
        :type element: char
        """
        # assignment(1) connect given node the next of the last linked node
        if node.next is None :
            node.next = Node(element, None)     # create next node

        else :
            self._link_last(element, node.next)      # visit next node


    def _link_next(self, index, element):
        """
        :type index: Int
        :type element: Char
        :param index
        :param element
        """
        next = self.get_node(index).next
        self.get_node(index).next = Node(element, next)

    def _unlink_first(self):
        # unlink first node of list
        x = self.head
        """:type : Node"""
        element = x.element
        self.head = x.next
        return element

    def _unlink_next(self, pred):
        """
        :type pred: Node
        :param pred:
        """
        x = pred.next
        element = x.element
        pred.next = x.next
        return element

    def _get_node(self, index, x):
        """
        :type index:int
        :type x:Node
        :param index:
        :param x:
        """
        # assignment(2) Get nth(index) node
        if index == 0:
            return x         # return current node

        elif index > 0:
            return self._get_node(index-1, x.next)       # return result of call _get_node


    def get_node(self, index):
        return self._get_node(index, self.head)

    def __len__(self):
        if self.head is None:
            return 0
        return len(self.head)

    def add(self, element, index=None):
        if index is None:
            if self.head is None:
                self._link_first(element)
            else:
                self._link_last(element, self.head)
            return

        if index < 0 or index > len(self):
            print "ERROR"
        elif index == 0:
            self._link_first(element)
        else:
            self._link_next(index-1, element)

    def remove(self, index):
        if index < 0 or index > len(self):
            print "ERROR"
        elif index == 0:
            return self._unlink_first()
        else:
            return self._unlink_next(self._get_node(index - 1, self.head))

    def __str__(self):
        if self.head is None:
            return "List is null"
        return str(self.head)

    def _reverse(self, x, pred):
        """
        :type x: Node
        :type pred: Node
        :param x:
        """
        if x.next is None :
            self.head = x
            x.next = pred

        else:
            self.head = x.next
            x.next = pred
            pred = x
            return  self._reverse(self.head, pred)
        # assignment(5)
        # Fill out, Use recursion

    def reverse(self):
        self._reverse(self.head, None)

    def iter_selection_sort(self):
        current_node = self.head
        compare_node = self.head
        min_node = self.head

        while current_node.next is not None:
            while compare_node is not None:
                if min_node.element > compare_node.element:
                    min_node = compare_node
                compare_node = compare_node.next
            current_node.element, min_node.element = min_node.element, current_node.element
            current_node = current_node.next
            compare_node = current_node
            min_node = current_node

    def selection_sort(self):
        self._selection(self.head)

    # Bonus Assignment
    def _selection(self, current_node):
    # Fill out, Use recursion
        if current_node.next is not None:
            self.compare(current_node, current_node.next, current_node)


    def compare(self, current_node, compare_node, min_node):
    # Fill out, Use recursion
        if min_node.element > compare_node.element:
            min_node = compare_node

        if compare_node.next is not None:
            self.compare(current_node, compare_node.next, min_node)

        if compare_node.next is None:
            current_node.element, min_node.element = min_node.element, current_node.element
            self._selection(current_node.next)


class Node(object):
    """
    :type element:char
    :type next: Node
    """

    def __init__(self, element, next):
        """
        :type element : char
        :type next : Node
        """
        self.element = element
        self.next = next

    def __str__(self):
        # assignment(3)
        if self.next is None:
            return str(self.element)    # Return string of self.element

        else:
            return str(self.element) +" " + str(self.next)    # Return self.element and string of next


    def __len__(self):
        if self.next is None :
            return 1
        else:
            return 1+len(self.next)
        # assignment(4) Return size of entire node


def test_recursion_linked_list():
    INPUT = ['a', 'b', 'c', 'd', 'e']
    test_RLL = RecursionLinkedList()
    for i in INPUT:
        test_RLL.add(i)
    print str(test_RLL)
    print "List size is " + str(len(test_RLL))
    test_RLL.add('z', 0)
    print "List size is " + str(len(test_RLL))
    print str(test_RLL)
    print test_RLL.get_node(4).element
    print test_RLL.remove(0)
    print str(test_RLL)
    test_RLL.reverse()
    print str(test_RLL)

def test_selection_sort():
    random_numbers=[]
    for i in range(10):
        random_numbers.append(random.randrange(0, 100))

    test_RSS = RecursionLinkedList()
    for i in random_numbers:
        test_RSS.add(i)
    print "List size is " + str(len(test_RSS))
    print str(test_RSS)
    #test_RLL.iter_selection_sort()
    test_RSS.selection_sort()
    print str(test_RSS)


test_recursion_linked_list()
test_selection_sort()
