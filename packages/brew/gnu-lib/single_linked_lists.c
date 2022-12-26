// ex‑gslist‑1.c
#include <glib.h>
#include <stdio.h>

void add_remove_data();
void add_double_data();
void pick_items();

int main(int argc, char** argv) {
  // GLIB functions of format: g_(container)_(function)

  printf("\n ******** new run: ********");
  printf("\n ** add and remove data:");
  add_remove_data();
  printf("\n ** add double datas:");
  add_double_data();
  printf("\n ** pick items from list:");
  pick_items();

  return 0;
}

// add, remove data:
void add_remove_data() {
  // create GSLIST:
  GSList* list = NULL;

  // get length of list:
  int len = g_slist_length(list);
  printf("\nThe list is now %d items long\n", len);

  // append list with one item:
  list = g_slist_append(list, "first");
  list = g_slist_append(list, "second");
  list = g_slist_append(list, "second");
  list = g_slist_append(list, "third");
  list = g_slist_append(list, "four");
  list = g_slist_append(list, "four");
  list = g_slist_append(list, "four");
  list = g_slist_append(list, "four");
  printf("The list is now %d items long\n", g_slist_length(list));

  // remove item from list:
  list = g_slist_remove(list, "third");
  printf("The list is now %d items long\n", g_slist_length(list));

  // remove all items of:
  list = g_slist_remove_all(list, "four");
  printf("The list is now %d items long\n", g_slist_length(list));

  // Free memory:
  g_slist_free(list);
}

void add_double_data() {
  GSList* list = NULL;
  double a = 2.345;
  double* pa = &a;
  list = g_slist_append(list, pa);
  list = g_slice_append(list, 3.5);
  printf("The last item is '%s'\n", g_list_last(list)->data);
  g_slist_free(list);
}

// pick data out of list:
void pick_items() {
  GSList* list = NULL;
  list = g_slist_append(list, "first");
  list = g_slist_append(list, "second");
  list = g_slist_append(list, "third");

  // get last item from list:
  char* last_item = g_list_last(list)->data;

  printf("The last item is '%s'\n", last_item);
  printf("The item at index '1' is '%s'\n", (char*)g_slist_nth(list, 1)->data);
  printf("Now the item at index '1' the easy way: '%s'\n",
         (char*)g_slist_nth_data(list, 1));
  printf("And the 'next' item after first item is '%s'\n",
         (char*)g_slist_next(list)->data);
  g_slist_free(list);
}