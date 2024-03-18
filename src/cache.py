import numpy as np
import random


class ItemCache:
    def __init__(self, min_counter=0, max_counter=50):
        self.cache = []
        self.min_counter = min_counter
        self.max_counter = max_counter

    def add_item_with_random_counter(self, item):
        counter = random.randint(self.min_counter, self.max_counter)
        self.cache.append((item, counter))

    def add_item_with_specific_counter(self, item, counter):
        self.cache.append((item, counter))

    def update_counters(self):
        """Decrements counters by 1 and removes items with a counter of 0, returning them."""
        removed_items = [item for item,
                         counter in self.cache if counter-1 <= 0]
        self.cache = [(item, counter-1)
                      for item, counter in self.cache if counter-1 > 0]
        return removed_items


# Example usage
if __name__ == "__main__":
    cache = ItemCache(min_counter=1, max_counter=5)
    iterations = 10

    for i in range(iterations):
        if i % 2 == 0:  # Example condition to add a random item
            cache.add_random_item()
        else:
            # Insert a specific item with a counter, here with a counter of 3 as an example
            specific_item = np.random.rand(5)
            cache.add_specific_item(specific_item, 3)

        removed_items = cache.update_counters()
        print(f"Iteration {i+1}: Removed {len(removed_items)} items")
