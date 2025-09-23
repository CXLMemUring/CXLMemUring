
#include "Conversion/CIRA.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/IR/DataLayout.h"

using namespace mlir;
using namespace mlir::cira;

// ============================================================================
// SCF disaggregation
//==============================================================================
class SCFWhileDisagg : public OpConversionPattern<scf::WhileOp> {
    using OpConversionPattern<scf::WhileOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(scf::WhileOp op, scf::WhileOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        // Prepare type mapping for before region signature change
        TypeConverter::SignatureConversion beforeArgs(op.getBeforeArguments().size());
        for (auto const &en : llvm::enumerate(adaptor.getOperands())) {
            beforeArgs.addInputs(en.index(), en.value().getType());
        }

        // Prepare for the after region signature conversion
        TypeConverter::SignatureConversion afterArgs(op.getAfterArguments().size());
        SmallVector<Type, 1> resultTypes;
        if (auto rts = op->getAttrOfType<mlir::ArrayAttr>("rel_types")) {
            for (Type en : rts.getAsValueRange<mlir::TypeAttr>())
                resultTypes.push_back(en);
        } else {
            for (Type en : op.getResultTypes())
                resultTypes.push_back(en);
        }
        for (auto const &en : llvm::enumerate(resultTypes)) {
            afterArgs.addInputs(en.index(), en.value());
        }

        // Start to transfer
        auto newWhileOp = rewriter.create<scf::WhileOp>(op.getLoc(), resultTypes, adaptor.getOperands());

        rewriter.inlineRegionBefore(op.getBefore(), newWhileOp.getBefore(), newWhileOp.getBefore().end());
        rewriter.applySignatureConversion(&newWhileOp.getBefore().front(), beforeArgs);

        rewriter.inlineRegionBefore(op.getAfter(), newWhileOp.getAfter(), newWhileOp.getAfter().end());
        rewriter.applySignatureConversion(&newWhileOp.getAfter().front(), afterArgs);

        rewriter.replaceOp(op, newWhileOp.getResults());
        return mlir::success();
    }
};

class SCFForOpDisagg : public OpConversionPattern<scf::ForOp> {
    using OpConversionPattern<scf::ForOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(scf::ForOp op, scf::ForOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        // Prepare for-loop body region signature conversion
        TypeConverter::SignatureConversion result(op.getBody()->getArguments().size());
        // induction variable is always the first
        result.addInputs(0, op.getInductionVar().getType());
        // push remapped operands (loop-carried vars)
        for (const auto &en : llvm::enumerate(adaptor.getInitArgs())) {
            result.addInputs(en.index() + 1, en.value().getType());
        }

        auto newForOp = rewriter.create<scf::ForOp>(op.getLoc(), adaptor.getLowerBound(), adaptor.getUpperBound(),
                                                    adaptor.getStep(), adaptor.getInitArgs());
        // forop builder will create block while building the region
        // delete it before moving original block into the new loop body
        // newForOp.getBody()->erase();
        rewriter.inlineRegionBefore(op.getBodyRegion(), newForOp.getBodyRegion(), newForOp.getBodyRegion().end());
        rewriter.applySignatureConversion(&newForOp.getBodyRegion().front(), result);
        rewriter.replaceOp(op, newForOp.getResults());
        return mlir::success();
    }
};

class SCFIfOpDisagg : public OpConversionPattern<scf::IfOp> {
    using OpConversionPattern<scf::IfOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(scf::IfOp op, scf::IfOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        SmallVector<Type, 4> relTypes;
        if (auto rts = op->getAttrOfType<mlir::ArrayAttr>("rel_types")) {
            for (Type en : rts.getAsValueRange<mlir::TypeAttr>())
                relTypes.push_back(en);
        } else {
            for (Type en : op.getResultTypes())
                relTypes.push_back(en);
        }

        auto newIf = rewriter.create<scf::IfOp>(op.getLoc(), relTypes, adaptor.getCondition(), true);
//        op->setAttr("remotable", IntegerAttr::get(IntegerType::get(getContext(), 32), 1));
        rewriter.eraseBlock(&newIf.getThenRegion().front());
        rewriter.inlineRegionBefore(op.getThenRegion(), newIf.getThenRegion(), newIf.getThenRegion().end());

        rewriter.eraseBlock(&newIf.getElseRegion().front());
        rewriter.inlineRegionBefore(op.getElseRegion(), newIf.getElseRegion(), newIf.getElseRegion().end());

        rewriter.replaceOp(op, newIf.getResults());
        return mlir::success();
    }
};

class SCFConditionOpDisagg : public OpConversionPattern<scf::ConditionOp> {
    using OpConversionPattern<scf::ConditionOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(scf::ConditionOp op, scf::ConditionOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<scf::ConditionOp>(op, adaptor.getCondition(), adaptor.getArgs());
        return mlir::success();
    }
};

class SCFYieldOpDisagg : public OpConversionPattern<scf::YieldOp> {
    using OpConversionPattern<scf::YieldOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(scf::YieldOp op, scf::YieldOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<scf::YieldOp>(op, adaptor.getOperands());
        return mlir::success();
    }
};

void mlir::cira::populateSCFCIRAPatterns(MLIRContext *ctx, RewritePatternSet &patterns) {
    patterns.add<
        /* SCF patterns */
        SCFWhileDisagg, SCFConditionOpDisagg, SCFYieldOpDisagg, SCFForOpDisagg, SCFIfOpDisagg>(ctx);
}
