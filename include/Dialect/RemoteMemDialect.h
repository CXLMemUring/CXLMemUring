#ifndef REMOTEMEM_DIALECT_H
#define REMOTEMEM_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir {
namespace cira {

class RemoteMemDialect : public Dialect {
public:
  explicit RemoteMemDialect(MLIRContext *context)
      : Dialect(getDialectNamespace(), context,
                TypeID::get<RemoteMemDialect>()) {
    initialize();
  }
  ~RemoteMemDialect() override = default;

  static StringRef getDialectNamespace() { return "cira"; }

  mlir::Attribute parseAttribute(mlir::DialectAsmParser &parser,
                               mlir::Type type) const override;
  void printAttribute(mlir::Attribute attr,
                    mlir::DialectAsmPrinter &printer) const override;

  void initialize();
  void registerTypes();
};

} // namespace cira
} // namespace mlir

#endif // REMOTEMEM_DIALECT_H 