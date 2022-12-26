#include <glib.h>
#include <stdio.h>

int main(int argc, char **argv) {
  GList *list = g_list_append(NULL, "Austin ");
  list = g_list_append(list, "Bowie ");
  list = g_list_append(list, "Bowie ");
  list = g_list_append(list, "Cheyenne ");
  printf("Here's the list: ");
  g_list_foreach(list, (GFunc)printf, NULL);
  printf("\nItem 'Bowie' is located at index %d\n",
         g_list_index(list, "Bowie "));
  printf("Item 'Dallas' is located at index %d\n",
         g_list_index(list, "Dallas"));
  GList *last = g_list_last(list);
  printf("Item 'Cheyenne' is located at index %d\n",
         g_list_position(list, last));
  g_list_free(list);

  GPtrArray *array;
  gchar *string1 = "one";
  gchar *string2 = "two";
  gchar *string3 = "three";

  array = g_ptr_array_new();
  g_ptr_array_add(array, (gpointer)string1);
  g_ptr_array_add(array, (gpointer)string2);
  g_ptr_array_add(array, (gpointer)string3);

  if (g_ptr_array_index(array, 0) != (gpointer)string1)
    g_print("ERROR: got %p instead of %p\n", g_ptr_array_index(array, 0),
            string1);

  g_ptr_array_free(array, TRUE);

  return 0;
}

/*
GLib bietet Unterstützung für
Basistypen
Standard-Makros
Typumwandlung
Konvertierung der Byte-Reihenfolge
Speicherreservierung
Warnungen und Zusicherungen
Nachricht-Protokollierung
Timer
Zeichenketten-Funktionen
Reguläre Ausdrücke
Hook-Funktionen
Lexikalisches Scannen
Parsen einer XML-Untermenge
Dynamisches Laden von Modulen
Threads
Speicher-Pools
Automatische Zeichenkettenvervollständigung
Typsystem (GType)
Datenstrukturen
Speicher-Chunks
Einfach und doppelt verkettete Listen
Hashtabellen
Dynamisch-wachsende Zeichenketten
Zeichenketten-Chunks
Felder
Balancierte Binärbäume
N-äre Bäume
Quarks
Relationen und Tupel
Caches
*/
