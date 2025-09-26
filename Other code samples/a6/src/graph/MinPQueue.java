package graph;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.NoSuchElementException;

/**
 * A min priority queue of distinct elements of type `KeyType` associated with (extrinsic) double
 * priorities, implemented using a binary heap paired with a hash table.
 */
public class MinPQueue<KeyType> {

    /**
     * Pairs an element `key` with its associated priority `priority`.
     */
    private record Entry<KeyType>(KeyType key, double priority) {
        // Note: This is equivalent to declaring a static nested class with fields `key` and
        //  `priority` and a corresponding constructor and observers, overriding `equals()` and
        //  `hashCode()` to depend on the fields, and overriding `toString()` to print their values.
        // https://docs.oracle.com/en/java/javase/17/language/records.html
    }

    /**
     * ArrayList representing a binary min-heap of element-priority pairs.  Satisfies
     * `heap.get(i).priority() >= heap.get((i-1)/2).priority()` for all `i` in `[1..heap.size())`.
     */
    private final ArrayList<Entry<KeyType>> heap;

    /**
     * Associates each element in the queue with its index in `heap`.  Satisfies
     * `heap.get(index.get(e)).key().equals(e)` if `e` is an element in the queue. Only maps
     * elements that are in the queue (`index.size() == heap.size()`).
     */
    private final Map<KeyType, Integer> index;

    // TODO 7 done: Write an assertInv() method that asserts that all of the class invariants are satisfied.
    public void assertInv() {
        for (int i = 0; i < heap.size(); i++) {
            assert (heap.get(i).priority() >= heap.get((i - 1) / 2).priority());
        }

        for (KeyType key : index.keySet()){
            assert (heap.get(index.get(key)).key().equals(key));
        }

        assert(index.size() == heap.size());
    }


    /**
     * Create an empty queue.
     */
    public MinPQueue() {
        index = new HashMap<>();
        heap = new ArrayList<>();
    }

    /**
     * Return whether this queue contains no elements.
     */
    public boolean isEmpty() {
        return heap.isEmpty();
    }

    /**
     * Return the number of elements contained in this queue.
     */
    public int size() {
        return heap.size();
    }

    /**
     * Return an element associated with the smallest priority in this queue.  This is the same
     * element that would be removed by a call to `remove()` (assuming no mutations in between).
     * Throws NoSuchElementException if this queue is empty.
     */
    public KeyType peek() {
        // Propagate exception from `List::getFirst()` if empty.
        return heap.getFirst().key();
    }

    /**
     * Return the minimum priority associated with an element in this queue.  Throws
     * NoSuchElementException if this queue is empty.
     */
    public double minPriority() {
        return heap.getFirst().priority();
    }

    /**
     * Swap the Entries at indices `i` and `j` in `heap`, updating `index` accordingly.  Requires
     * {@code 0 <= i,j < heap.size()}.
     */
    private void swap(int i, int j) {
        // TODO 8a done but unchecked: Implement this method according to its specification
        assert ((0<=i)&&(i<heap.size())&&(0<=j)&&(j<heap.size()));

        Entry<KeyType> valueI = heap.get(i);
        Entry<KeyType> valueJ = heap.get(j);

        heap.set(i,valueJ);
        heap.set(j,valueI);

        index.put(valueI.key(), j);
        index.put(valueJ.key(), i);

        //assertInv(); //FIXME Figure out a way to assert this
    }

    // TODO 8b Return when bubbling up or down is needed: Implement private helper methods for bubbling entries up and down in the heap.
    //  Their interfaces are up to you, but you must write precise specifications.
    private void bubbleUp(int i){
        // im deferring this until I need the method, so I know what form it must take
        if (heap.get(i).priority()<heap.get((i - 1) / 2).priority()){
            swap(i, (i - 1) / 2);
            bubbleUp((i - 1) / 2);
        }
        //assertInv(); //FIXME Figure out a way to assert this
    }
    private void bubbleDown(int i){
        // im deferring this until I need the method, so I know what form it must take
        int size = heap.size();
        int leftIndex  = 2 * i + 1;
        int rightIndex = 2 * i + 2;
        if (leftIndex >= size) {
            assertInv();
            return;
        }

        int smallerChildIndex = leftIndex;
        if (rightIndex < size) {
            if (heap.get(rightIndex).priority() < heap.get(leftIndex).priority()) {
                smallerChildIndex = rightIndex;
            }
        }

        if (heap.get(i).priority()>heap.get(smallerChildIndex).priority()){
            swap(i, smallerChildIndex);
            bubbleDown(smallerChildIndex);
        }
        assertInv();
    }


    /**
     * Add element `key` to this queue, associated with priority `priority`.  Requires `key` is not
     * contained in this queue.
     */
    private void add(KeyType key, double priority) {
        // TODO 9a done: Implement this method according to its specification
        if (!index.containsKey(key)){
            int i = heap.size();
            heap.add(new Entry<>(key, priority));
            index.put(key, i);
            bubbleUp(i);
        }
        assertInv();
    }

    /**
     * Change the priority associated with element `key` to `priority`.  Requires that `key` is
     * contained in this queue.
     */
    private void update(KeyType key, double priority) {
        assert index.containsKey(key);
        // TODO 9b done but i dont trust it: Implement this method according to its specification
        int i = index.get(key);

        heap.set(i, new Entry<>(key, priority));
        bubbleUp(i);
        bubbleDown(i);

        assertInv();
    }

    /**
     * If `key` is already contained in this queue, change its associated priority to `priority`.
     * Otherwise, add it to this queue with that priority.
     */
    public void addOrUpdate(KeyType key, double priority) {
        if (!index.containsKey(key)) {
            add(key, priority);
        } else {
            update(key, priority);
        }
        assertInv();
    }

    /**
     * Remove and return the element associated with the smallest priority in this queue.  If
     * multiple elements are tied for the smallest priority, an arbitrary one will be removed.
     * Throws NoSuchElementException if this queue is empty.
     */
    public KeyType remove() {
        // TODO 9c done: Implement this method according to its specification
        //throw new UnsupportedOperationException();
        int finalIndex = heap.size()-1;
        swap(0, finalIndex);
        Entry<KeyType> element = heap.remove(finalIndex);
        index.remove(element.key);
        bubbleDown(0);
        assertInv();
        return element.key;

    }

}
