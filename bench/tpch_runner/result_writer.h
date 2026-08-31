#ifndef CIRA_TPCH_RESULT_WRITER_H
#define CIRA_TPCH_RESULT_WRITER_H

#include <monetdbe.h>
#include <stdbool.h>

int tpch_write_result(monetdbe_database database, monetdbe_result *result,
                      const char *output_directory, int query_id,
                      bool ordered);

#endif
