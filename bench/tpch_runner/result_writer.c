#include "result_writer.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int write_json_string(FILE *stream, const char *text) {
  if (fputc('"', stream) == EOF)
    return -1;
  for (const unsigned char *cursor = (const unsigned char *)text; *cursor;
       ++cursor) {
    switch (*cursor) {
    case '"':
      if (fputs("\\\"", stream) == EOF)
        return -1;
      break;
    case '\\':
      if (fputs("\\\\", stream) == EOF)
        return -1;
      break;
    case '\n':
      if (fputs("\\n", stream) == EOF)
        return -1;
      break;
    case '\r':
      if (fputs("\\r", stream) == EOF)
        return -1;
      break;
    case '\t':
      if (fputs("\\t", stream) == EOF)
        return -1;
      break;
    default:
      if (*cursor < 0x20) {
        if (fprintf(stream, "\\u%04x", *cursor) < 0)
          return -1;
      } else if (fputc(*cursor, stream) == EOF) {
        return -1;
      }
    }
  }
  return fputc('"', stream) == EOF ? -1 : 0;
}

static int write_tsv_field(FILE *stream, const char *text) {
  bool quote = strpbrk(text, "\t\r\n\"") != NULL;
  if (quote && fputc('"', stream) == EOF)
    return -1;
  for (const char *cursor = text; *cursor; ++cursor) {
    if (*cursor == '"' && fputc('"', stream) == EOF)
      return -1;
    if (fputc(*cursor, stream) == EOF)
      return -1;
  }
  return quote && fputc('"', stream) == EOF ? -1 : 0;
}

static void unsigned_128_string(unsigned __int128 value, char *output) {
  char reverse[64];
  size_t count = 0;
  do {
    reverse[count++] = (char)('0' + value % 10);
    value /= 10;
  } while (value);
  for (size_t index = 0; index < count; ++index)
    output[index] = reverse[count - index - 1];
  output[count] = '\0';
}

static void format_integer(__int128 value, unsigned scale, char *output,
                           size_t output_size) {
  bool negative = value < 0;
  unsigned __int128 magnitude = negative
                                    ? (unsigned __int128)(-(value + 1)) + 1
                                    : (unsigned __int128)value;
  char digits[64];
  unsigned_128_string(magnitude, digits);
  size_t count = strlen(digits);
  char *cursor = output;
  size_t remaining = output_size;
#define APPEND(...)                                                             \
  do {                                                                          \
    int written = snprintf(cursor, remaining, __VA_ARGS__);                     \
    if (written < 0 || (size_t)written >= remaining) {                          \
      output[0] = '\0';                                                         \
      return;                                                                   \
    }                                                                           \
    cursor += written;                                                          \
    remaining -= (size_t)written;                                               \
  } while (0)
  if (negative)
    APPEND("-");
  if (scale == 0) {
    APPEND("%s", digits);
  } else if (count <= scale) {
    APPEND("0.");
    for (size_t zeros = scale - count; zeros; --zeros)
      APPEND("0");
    APPEND("%s", digits);
  } else {
    APPEND("%.*s.%s", (int)(count - scale), digits, digits + count - scale);
  }
#undef APPEND
}

static bool is_decimal(const monetdbe_column *column) {
  const char *name = column->sql_type.name;
  return name && (strcmp(name, "decimal") == 0 || strcmp(name, "numeric") == 0);
}

static void column_type_name(const monetdbe_column *column, char *output,
                             size_t output_size) {
  if (is_decimal(column)) {
    snprintf(output, output_size, "%s(%u,%u)", column->sql_type.name,
             column->sql_type.digits, column->sql_type.scale);
    return;
  }
  if (column->sql_type.name && *column->sql_type.name) {
    snprintf(output, output_size, "%s", column->sql_type.name);
    return;
  }
  static const char *fallback[] = {
      "boolean", "tinyint", "smallint", "integer", "bigint",
#ifdef HAVE_HGE
      "hugeint",
#endif
      "size_t", "real", "double", "varchar", "blob", "date", "time",
      "timestamp", "unknown"};
  size_t index = (size_t)column->type;
  snprintf(output, output_size, "%s",
           index < sizeof(fallback) / sizeof(fallback[0]) ? fallback[index]
                                                          : "unknown");
}

#define TYPED_COLUMN(name) ((monetdbe_column_##name *)column)
#define COLUMN_NULL(name, row)                                                  \
  (TYPED_COLUMN(name)->is_null && TYPED_COLUMN(name)->is_null(                  \
                                      TYPED_COLUMN(name)->data + (row)))

static int write_cell(FILE *stream, monetdbe_column *column, size_t row) {
  char value[256];
  bool null_value = false;
  switch (column->type) {
  case monetdbe_bool:
    null_value = COLUMN_NULL(bool, row);
    if (!null_value)
      snprintf(value, sizeof(value), "%s",
               TYPED_COLUMN(bool)->data[row] ? "true" : "false");
    break;
  case monetdbe_int8_t:
    null_value = COLUMN_NULL(int8_t, row);
    if (!null_value)
      format_integer(TYPED_COLUMN(int8_t)->data[row],
                     is_decimal(column) ? column->sql_type.scale : 0, value,
                     sizeof(value));
    break;
  case monetdbe_int16_t:
    null_value = COLUMN_NULL(int16_t, row);
    if (!null_value)
      format_integer(TYPED_COLUMN(int16_t)->data[row],
                     is_decimal(column) ? column->sql_type.scale : 0, value,
                     sizeof(value));
    break;
  case monetdbe_int32_t:
    null_value = COLUMN_NULL(int32_t, row);
    if (!null_value)
      format_integer(TYPED_COLUMN(int32_t)->data[row],
                     is_decimal(column) ? column->sql_type.scale : 0, value,
                     sizeof(value));
    break;
  case monetdbe_int64_t:
    null_value = COLUMN_NULL(int64_t, row);
    if (!null_value)
      format_integer(TYPED_COLUMN(int64_t)->data[row],
                     is_decimal(column) ? column->sql_type.scale : 0, value,
                     sizeof(value));
    break;
#ifdef HAVE_HGE
  case monetdbe_int128_t:
    null_value = COLUMN_NULL(int128_t, row);
    if (!null_value)
      format_integer(TYPED_COLUMN(int128_t)->data[row],
                     is_decimal(column) ? column->sql_type.scale : 0, value,
                     sizeof(value));
    break;
#endif
  case monetdbe_size_t:
    null_value = COLUMN_NULL(size_t, row);
    if (!null_value)
      snprintf(value, sizeof(value), "%zu", TYPED_COLUMN(size_t)->data[row]);
    break;
  case monetdbe_float:
    null_value = COLUMN_NULL(float, row);
    if (!null_value)
      snprintf(value, sizeof(value), "%.9g", TYPED_COLUMN(float)->data[row]);
    break;
  case monetdbe_double:
    null_value = COLUMN_NULL(double, row);
    if (!null_value)
      snprintf(value, sizeof(value), "%.17g", TYPED_COLUMN(double)->data[row]);
    break;
  case monetdbe_str:
    null_value = COLUMN_NULL(str, row);
    if (!null_value)
      return write_tsv_field(stream, TYPED_COLUMN(str)->data[row]);
    break;
  case monetdbe_date: {
    null_value = COLUMN_NULL(date, row);
    if (!null_value) {
      monetdbe_data_date date = TYPED_COLUMN(date)->data[row];
      snprintf(value, sizeof(value), "%04d-%02u-%02u", date.year, date.month,
               date.day);
    }
    break;
  }
  case monetdbe_time: {
    null_value = COLUMN_NULL(time, row);
    if (!null_value) {
      monetdbe_data_time time = TYPED_COLUMN(time)->data[row];
      snprintf(value, sizeof(value), "%02u:%02u:%02u.%03u", time.hours,
               time.minutes, time.seconds, time.ms);
    }
    break;
  }
  case monetdbe_timestamp: {
    null_value = COLUMN_NULL(timestamp, row);
    if (!null_value) {
      monetdbe_data_timestamp timestamp = TYPED_COLUMN(timestamp)->data[row];
      snprintf(value, sizeof(value), "%04d-%02u-%02uT%02u:%02u:%02u.%03u",
               timestamp.date.year, timestamp.date.month, timestamp.date.day,
               timestamp.time.hours, timestamp.time.minutes,
               timestamp.time.seconds, timestamp.time.ms);
    }
    break;
  }
  case monetdbe_blob: {
    null_value = COLUMN_NULL(blob, row);
    if (!null_value) {
      monetdbe_data_blob blob = TYPED_COLUMN(blob)->data[row];
      if (fputs("0x", stream) == EOF)
        return -1;
      for (size_t index = 0; index < blob.size; ++index)
        if (fprintf(stream, "%02x", (unsigned char)blob.data[index]) < 0)
          return -1;
      return 0;
    }
    break;
  }
  default:
    errno = EINVAL;
    return -1;
  }
  return write_tsv_field(stream, null_value ? "\\N" : value);
}

int tpch_write_result(monetdbe_database database, monetdbe_result *result,
                      const char *output_directory, int query_id,
                      bool ordered) {
  char tsv_path[4096];
  char json_path[4096];
  if (snprintf(tsv_path, sizeof(tsv_path), "%s/q%02d.tsv", output_directory,
               query_id) >= (int)sizeof(tsv_path) ||
      snprintf(json_path, sizeof(json_path), "%s/q%02d.json", output_directory,
               query_id) >= (int)sizeof(json_path)) {
    errno = ENAMETOOLONG;
    return -1;
  }

  monetdbe_column **columns = calloc(result->ncols, sizeof(*columns));
  if (!columns)
    return -1;
  for (size_t column = 0; column < result->ncols; ++column) {
    char *error = monetdbe_result_fetch(result, &columns[column], column);
    if (error) {
      fprintf(stderr, "monetdbe_result_fetch: %s\n", error);
      free(columns);
      return -1;
    }
  }

  FILE *tsv = fopen(tsv_path, "w");
  if (!tsv) {
    free(columns);
    return -1;
  }
  int failed = 0;
  for (monetdbe_cnt row = 0; row < result->nrows && !failed; ++row) {
    for (size_t column = 0; column < result->ncols; ++column) {
      if (column && fputc('\t', tsv) == EOF)
        failed = 1;
      if (!failed && write_cell(tsv, columns[column], (size_t)row) != 0)
        failed = 1;
    }
    if (!failed && fputc('\n', tsv) == EOF)
      failed = 1;
  }
  if (fclose(tsv) != 0)
    failed = 1;

  FILE *json = failed ? NULL : fopen(json_path, "w");
  if (!json)
    failed = 1;
  if (!failed) {
    fprintf(json,
            "{\n  \"schema_version\": 1,\n  \"query_id\": %d,\n"
            "  \"ordered\": %s,\n  \"row_count\": %" PRId64
            ",\n  \"columns\": [\n",
            query_id, ordered ? "true" : "false", result->nrows);
    for (size_t column = 0; column < result->ncols; ++column) {
      char type[128];
      column_type_name(columns[column], type, sizeof(type));
      fputs(column ? ",\n    {\"name\": " : "    {\"name\": ", json);
      write_json_string(json, columns[column]->name ? columns[column]->name : "");
      fputs(", \"type\": ", json);
      write_json_string(json, type);
      fputc('}', json);
    }
    fputs("\n  ]\n}\n", json);
    if (fclose(json) != 0)
      failed = 1;
  }
  free(columns);
  (void)database;
  return failed ? -1 : 0;
}
