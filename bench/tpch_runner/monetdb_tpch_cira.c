#include "result_writer.h"
#include "cira_shared_arena.h"

#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

typedef void *(*gdk_cira_alloc_fn)(size_t, size_t);
typedef void (*gdk_cira_free_fn)(void *, size_t);
extern void GDKset_cira_heap_hooks(gdk_cira_alloc_fn, gdk_cira_free_fn)
    __attribute__((weak));

extern void m5_work_begin(uint64_t, uint64_t) __attribute__((weak));
extern void m5_work_end(uint64_t, uint64_t) __attribute__((weak));
extern void cira_offload_set_query_context(uint32_t);

static cira_shared_arena_t *shared_arena;

static void *arena_alloc(size_t bytes, size_t alignment) {
  uint64_t offset = 0;
  if (!shared_arena ||
      cira_shared_arena_alloc(shared_arena, bytes, alignment, &offset) !=
          CIRA_ARENA_OK)
    return NULL;
  return cira_shared_arena_from_offset(shared_arena, offset, bytes, alignment);
}

static void arena_free(void *pointer, size_t bytes) {
  (void)pointer;
  (void)bytes;
}

static char *read_file(const char *path) {
  FILE *stream = fopen(path, "rb");
  if (!stream)
    return NULL;
  if (fseek(stream, 0, SEEK_END) != 0) {
    fclose(stream);
    return NULL;
  }
  long size = ftell(stream);
  if (size < 0 || fseek(stream, 0, SEEK_SET) != 0) {
    fclose(stream);
    return NULL;
  }
  char *content = malloc((size_t)size + 1);
  if (!content) {
    fclose(stream);
    return NULL;
  }
  if (fread(content, 1, (size_t)size, stream) != (size_t)size) {
    free(content);
    fclose(stream);
    return NULL;
  }
  content[size] = '\0';
  fclose(stream);
  return content;
}

static int execute(monetdbe_database database, char *sql,
                   monetdbe_result **result) {
  char *error = monetdbe_query(database, sql, result, NULL);
  if (error) {
    fprintf(stderr, "MonetDB query failed: %s\nSQL: %.400s\n", error, sql);
    return -1;
  }
  return 0;
}

static int execute_file(monetdbe_database database, const char *path,
                        monetdbe_result **result) {
  char *sql = read_file(path);
  if (!sql) {
    fprintf(stderr, "cannot read SQL file %s: %s\n", path, strerror(errno));
    return -1;
  }
  int status = execute(database, sql, result);
  free(sql);
  return status;
}

static int execute_schema(monetdbe_database database, const char *path) {
  char *schema = read_file(path);
  if (!schema)
    return -1;
  char *start = schema;
  bool quoted = false;
  int status = 0;
  for (char *cursor = schema; *cursor; ++cursor) {
    if (*cursor == '\'' && (cursor == schema || cursor[-1] != '\\'))
      quoted = !quoted;
    if (*cursor != ';' || quoted)
      continue;
    *cursor = '\0';
    if (strspn(start, " \t\r\n") != strlen(start) &&
        execute(database, start, NULL) != 0) {
      status = -1;
      break;
    }
    start = cursor + 1;
  }
  free(schema);
  return status;
}

static char *sql_quote_path(const char *path) {
  size_t length = strlen(path);
  size_t quotes = 0;
  for (const char *cursor = path; *cursor; ++cursor)
    quotes += *cursor == '\'';
  char *quoted = malloc(length + quotes + 1);
  if (!quoted)
    return NULL;
  char *output = quoted;
  for (const char *cursor = path; *cursor; ++cursor) {
    *output++ = *cursor;
    if (*cursor == '\'')
      *output++ = '\'';
  }
  *output = '\0';
  return quoted;
}

struct table_count {
  const char *name;
  int64_t rows;
};

static const struct table_count table_counts[] = {
    {"region", 5},       {"nation", 25},     {"supplier", 100},
    {"customer", 1500}, {"part", 2000},     {"partsupp", 8000},
    {"orders", 15000},  {"lineitem", 60175},
};

static int load_tables(monetdbe_database database, const char *data) {
  for (size_t index = 0; index < sizeof(table_counts) / sizeof(table_counts[0]);
       ++index) {
    char path[4096];
    if (snprintf(path, sizeof(path), "%s/%s.tbl", data,
                 table_counts[index].name) >= (int)sizeof(path))
      return -1;
    char *quoted = sql_quote_path(path);
    if (!quoted)
      return -1;
    char sql[8192];
    int written = snprintf(
        sql, sizeof(sql),
        "COPY %" PRId64 " RECORDS INTO %s FROM E'%s' USING DELIMITERS '|', "
        "E'|\\n'",
        table_counts[index].rows, table_counts[index].name, quoted);
    free(quoted);
    if (written < 0 || written >= (int)sizeof(sql) ||
        execute(database, sql, NULL) != 0)
      return -1;
  }
  return 0;
}

static int64_t count_result(monetdbe_result *result) {
  monetdbe_column *column = NULL;
  char *error = monetdbe_result_fetch(result, &column, 0);
  if (error || result->nrows != 1)
    return -1;
  switch (column->type) {
  case monetdbe_int32_t:
    return ((monetdbe_column_int32_t *)column)->data[0];
  case monetdbe_int64_t:
    return ((monetdbe_column_int64_t *)column)->data[0];
#ifdef HAVE_HGE
  case monetdbe_int128_t:
    return (int64_t)((monetdbe_column_int128_t *)column)->data[0];
#endif
  default:
    return -1;
  }
}

static int verify_counts(monetdbe_database database) {
  for (size_t index = 0; index < sizeof(table_counts) / sizeof(table_counts[0]);
       ++index) {
    char sql[256];
    snprintf(sql, sizeof(sql), "SELECT COUNT(*) FROM %s", table_counts[index].name);
    monetdbe_result *result = NULL;
    if (execute(database, sql, &result) != 0)
      return -1;
    int64_t actual = count_result(result);
    char *error = monetdbe_cleanup_result(database, result);
    if (error || actual != table_counts[index].rows) {
      fprintf(stderr, "%s row count: expected %" PRId64 ", got %" PRId64 "\n",
              table_counts[index].name, table_counts[index].rows, actual);
      return -1;
    }
  }
  return 0;
}

static int read_ordered(const char *queries, int query_id, bool *ordered) {
  char path[4096];
  snprintf(path, sizeof(path), "%s/q%02d.ordered", queries, query_id);
  char *text = read_file(path);
  if (!text || (text[0] != '0' && text[0] != '1')) {
    free(text);
    fprintf(stderr, "invalid ordering contract: %s\n", path);
    return -1;
  }
  *ordered = text[0] == '1';
  free(text);
  return 0;
}

static int run_query(monetdbe_database database, const char *queries,
                     const char *output, int query_id, uint64_t mode_tag) {
  char query_path[4096];
  char setup_path[4096];
  char cleanup_path[4096];
  snprintf(query_path, sizeof(query_path), "%s/q%02d.sql", queries, query_id);
  snprintf(setup_path, sizeof(setup_path), "%s/q%02d-setup.sql", queries,
           query_id);
  snprintf(cleanup_path, sizeof(cleanup_path), "%s/q%02d-cleanup.sql", queries,
           query_id);
  cira_offload_set_query_context((uint32_t)query_id);
  FILE *probe = fopen(setup_path, "r");
  if (probe) {
    fclose(probe);
    if (execute_file(database, setup_path, NULL) != 0) {
      cira_offload_set_query_context(0);
      return -1;
    }
  }
  bool ordered = false;
  if (read_ordered(queries, query_id, &ordered) != 0) {
    cira_offload_set_query_context(0);
    return -1;
  }
  if (m5_work_begin)
    m5_work_begin((uint64_t)query_id, mode_tag);
  monetdbe_result *result = NULL;
  int status = execute_file(database, query_path, &result);
  if (m5_work_end)
    m5_work_end((uint64_t)query_id, mode_tag);
  if (status == 0 && (!result || tpch_write_result(database, result, output,
                                                   query_id, ordered) != 0))
    status = -1;
  if (result) {
    char *error = monetdbe_cleanup_result(database, result);
    if (error) {
      fprintf(stderr, "MonetDB cleanup failed: %s\n", error);
      status = -1;
    }
  }
  probe = fopen(cleanup_path, "r");
  if (probe) {
    fclose(probe);
    if (execute_file(database, cleanup_path, NULL) != 0)
      status = -1;
  }
  if (status == 0)
    printf("Q%02d PASS rows written\n", query_id);
  cira_offload_set_query_context(0);
  return status;
}

static const char *argument(int argc, char **argv, const char *name) {
  for (int index = 1; index + 1 < argc; ++index)
    if (strcmp(argv[index], name) == 0)
      return argv[index + 1];
  return NULL;
}

int main(int argc, char **argv) {
  const char *data = argument(argc, argv, "--data");
  const char *queries = argument(argc, argv, "--queries");
  const char *schema = argument(argc, argv, "--schema");
  const char *output = argument(argc, argv, "--output");
  const char *mode = argument(argc, argv, "--mode");
  const char *query = argument(argc, argv, "--query");
  if (!data || !queries || !schema || !output || !mode || !query ||
      (strcmp(mode, "baseline") != 0 && strcmp(mode, "forced-debug") != 0)) {
    fprintf(stderr,
            "usage: %s --data DIR --queries DIR --schema FILE --output DIR "
            "--mode baseline|forced-debug --query all|1..22\n",
            argv[0]);
    return 2;
  }

  uint64_t mode_tag = strcmp(mode, "forced-debug") == 0 ? 1 : 0;
  if (setenv("CIRA_OFFLOAD_MODE", mode, 1) != 0) {
    perror("setenv CIRA_OFFLOAD_MODE");
    return 1;
  }
  if (!getenv("CIRA_OFFLOAD_TELEMETRY")) {
    char telemetry_path[4096];
    int length = snprintf(telemetry_path, sizeof(telemetry_path),
                          "%s/telemetry.jsonl", output);
    if (length < 0 || (size_t)length >= sizeof(telemetry_path) ||
        setenv("CIRA_OFFLOAD_TELEMETRY", telemetry_path, 1) != 0) {
      fprintf(stderr, "cannot configure offload telemetry path\n");
      return 1;
    }
  }
  if (mode_tag) {
    if (!GDKset_cira_heap_hooks) {
      fprintf(stderr,
              "forced-debug requires a MonetDB build with CIRA heap hooks\n");
      return 1;
    }
    shared_arena = cira_shared_arena_default();
    if (!shared_arena) {
      fprintf(stderr, "forced-debug requires a configured CIRA shared arena\n");
      return 1;
    }
    GDKset_cira_heap_hooks(arena_alloc, arena_free);
  } else if (GDKset_cira_heap_hooks) {
    GDKset_cira_heap_hooks(NULL, NULL);
  }

  monetdbe_database database = NULL;
  monetdbe_options options = {0};
  options.nr_threads = 1;
  int open_status = monetdbe_open(&database, NULL, &options);
  if (open_status != 0) {
    const char *error = database ? monetdbe_error(database) : NULL;
    fprintf(stderr, "monetdbe_open failed (%d)%s%s\n", open_status,
            error ? ": " : "", error ? error : "");
    return 1;
  }

  int status = execute_schema(database, schema);
  if (status == 0)
    status = load_tables(database, data);
  if (status == 0)
    status = verify_counts(database);
  if (status == 0) {
    if (strcmp(query, "all") == 0) {
      for (int query_id = 1; query_id <= 22 && status == 0; ++query_id)
        status = run_query(database, queries, output, query_id, mode_tag);
    } else {
      char *end = NULL;
      long query_id = strtol(query, &end, 10);
      if (!end || *end || query_id < 1 || query_id > 22)
        status = -1;
      else
        status = run_query(database, queries, output, (int)query_id, mode_tag);
    }
  }
  if (monetdbe_close(database) != 0)
    status = -1;
  return status == 0 ? 0 : 1;
}
