; ModuleID = 'naive_promotable_accumulate'
source_filename = "<string>"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

@.const.naive_promotable_accumulate = internal constant [28 x i8] c"naive_promotable_accumulate\00"
@_ZN08NumbaEnv8__main__27naive_promotable_accumulateB2v2B40c8tJTIeFIjxB2IKSgI4CrvQClQZ6FcpCShpgEU0AE5ArrayIyLi1E1C7mutable7alignedEyyy = common local_unnamed_addr global ptr null
@".const.missing Environment: _ZN08NumbaEnv8__main__27naive_promotable_accumulateB2v2B40c8tJTIeFIjxB2IKSgI4CrvQClQZ6FcpCShpgEU0AE5ArrayIyLi1E1C7mutable7alignedEyyy" = internal constant [155 x i8] c"missing Environment: _ZN08NumbaEnv8__main__27naive_promotable_accumulateB2v2B40c8tJTIeFIjxB2IKSgI4CrvQClQZ6FcpCShpgEU0AE5ArrayIyLi1E1C7mutable7alignedEyyy\00"
@PyExc_TypeError = external global i8
@".const.can't unbox array from PyObject into native value.  The object maybe of a different type" = internal constant [89 x i8] c"can't unbox array from PyObject into native value.  The object maybe of a different type\00"
@PyExc_RuntimeError = external global i8

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define noundef i32 @_ZN8__main__27naive_promotable_accumulateB2v2B40c8tJTIeFIjxB2IKSgI4CrvQClQZ6FcpCShpgEU0AE5ArrayIyLi1E1C7mutable7alignedEyyy(ptr noalias writeonly captures(none) %retptr, ptr noalias readnone captures(none) %excinfo, ptr readnone captures(none) %arg.buffer.0, ptr readnone captures(none) %arg.buffer.1, i64 %arg.buffer.2, i64 %arg.buffer.3, ptr captures(none) %arg.buffer.4, i64 %arg.buffer.5.0, i64 %arg.buffer.6.0, i64 %arg.slot, i64 %arg.iters, i64 %arg.seed) local_unnamed_addr #0 {
B0.endif:
  %.10778.not = icmp slt i64 %arg.iters, 1
  br i1 %.10778.not, label %B144, label %iter.check

iter.check:                                       ; preds = %B0.endif
  %const38 = bitcast i64 -7046029254386353131 to i64
  %.175 = getelementptr i64, ptr %arg.buffer.4, i64 %arg.slot
  %.175.promoted = load i64, ptr %.175, align 8
  %min.iters.check = icmp ult i64 %arg.iters, 4
  br i1 %min.iters.check, label %B34.preheader, label %vector.main.loop.iter.check

vector.main.loop.iter.check:                      ; preds = %iter.check
  %const = bitcast i64 9223372036854775776 to i64
  %min.iters.check1 = icmp ult i64 %arg.iters, 32
  br i1 %min.iters.check1, label %vec.epilog.ph, label %vector.ph

vector.ph:                                        ; preds = %vector.main.loop.iter.check
  %n.vec = and i64 %arg.iters, %const
  %0 = mul i64 %n.vec, %const38
  %1 = add i64 %arg.seed, %0
  %2 = insertelement <8 x i64> <i64 poison, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0>, i64 %.175.promoted, i64 0
  %broadcast.splatinsert = insertelement <8 x i64> poison, i64 %arg.seed, i64 0
  %broadcast.splat = shufflevector <8 x i64> %broadcast.splatinsert, <8 x i64> poison, <8 x i32> zeroinitializer
  %induction = add <8 x i64> %broadcast.splat, <i64 0, i64 -7046029254386353131, i64 4354685564936845354, i64 -2691343689449507777, i64 8709371129873690708, i64 1663341875487337577, i64 -5382687378899015554, i64 6018027440424182931>
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %lsr.iv36 = phi i64 [ %lsr.iv.next37, %vector.body ], [ %n.vec, %vector.ph ]
  %vec.phi = phi <8 x i64> [ %2, %vector.ph ], [ %15, %vector.body ]
  %vec.phi2 = phi <8 x i64> [ zeroinitializer, %vector.ph ], [ %16, %vector.body ]
  %vec.phi3 = phi <8 x i64> [ zeroinitializer, %vector.ph ], [ %17, %vector.body ]
  %vec.phi4 = phi <8 x i64> [ zeroinitializer, %vector.ph ], [ %18, %vector.body ]
  %vec.ind = phi <8 x i64> [ %induction, %vector.ph ], [ %vec.ind.next, %vector.body ]
  %3 = add <8 x i64> %vec.ind, splat (i64 -7046029254386353131)
  %4 = add <8 x i64> %vec.ind, splat (i64 -8074031068348523331)
  %5 = add <8 x i64> %vec.ind, splat (i64 -9102032882310693531)
  %6 = add <8 x i64> %vec.ind, splat (i64 8316709377436687885)
  %7 = lshr <8 x i64> %3, splat (i64 58)
  %8 = lshr <8 x i64> %4, splat (i64 58)
  %9 = lshr <8 x i64> %5, splat (i64 58)
  %10 = lshr <8 x i64> %6, splat (i64 58)
  %11 = add <8 x i64> %vec.phi, splat (i64 1)
  %12 = add <8 x i64> %vec.phi2, splat (i64 1)
  %13 = add <8 x i64> %vec.phi3, splat (i64 1)
  %14 = add <8 x i64> %vec.phi4, splat (i64 1)
  %15 = add <8 x i64> %11, %7
  %16 = add <8 x i64> %12, %8
  %17 = add <8 x i64> %13, %9
  %18 = add <8 x i64> %14, %10
  %vec.ind.next = add <8 x i64> %vec.ind, splat (i64 -4112007255848680800)
  %lsr.iv.next37 = add nsw i64 %lsr.iv36, -32
  %19 = icmp eq i64 %lsr.iv.next37, 0
  br i1 %19, label %middle.block, label %vector.body, !llvm.loop !0

middle.block:                                     ; preds = %vector.body
  %bin.rdx = add <8 x i64> %16, %15
  %bin.rdx5 = add <8 x i64> %17, %bin.rdx
  %bin.rdx6 = add <8 x i64> %18, %bin.rdx5
  %rdx.shuf = shufflevector <8 x i64> %bin.rdx6, <8 x i64> poison, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison>
  %bin.rdx39 = add <8 x i64> %bin.rdx6, %rdx.shuf
  %rdx.shuf40 = shufflevector <8 x i64> %bin.rdx39, <8 x i64> poison, <8 x i32> <i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %bin.rdx41 = add <8 x i64> %bin.rdx39, %rdx.shuf40
  %rdx.shuf42 = shufflevector <8 x i64> %bin.rdx41, <8 x i64> poison, <8 x i32> <i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %bin.rdx43 = add <8 x i64> %bin.rdx41, %rdx.shuf42
  %20 = extractelement <8 x i64> %bin.rdx43, i32 0
  %cmp.n = icmp eq i64 %arg.iters, %n.vec
  br i1 %cmp.n, label %B30.B144_crit_edge, label %vec.epilog.iter.check

vec.epilog.iter.check:                            ; preds = %middle.block
  %21 = mul i64 %n.vec, %const38
  %ind.end = add i64 %arg.seed, %21
  %ind.end21 = and i64 %arg.iters, 31
  %22 = and i64 %arg.iters, 28
  %min.epilog.iters.check = icmp eq i64 %22, 0
  br i1 %min.epilog.iters.check, label %B34.preheader, label %vec.epilog.ph, !prof !3

vec.epilog.ph:                                    ; preds = %vector.main.loop.iter.check, %vec.epilog.iter.check
  %vec.epilog.resume.val = phi i64 [ %n.vec, %vec.epilog.iter.check ], [ 0, %vector.main.loop.iter.check ]
  %bc.merge.rdx = phi i64 [ %20, %vec.epilog.iter.check ], [ %.175.promoted, %vector.main.loop.iter.check ]
  %bc.resume.val = phi i64 [ %1, %vec.epilog.iter.check ], [ %arg.seed, %vector.main.loop.iter.check ]
  %const_mat = add i64 %const, 28
  %n.vec8 = and i64 %arg.iters, %const_mat
  %23 = mul i64 %n.vec8, %const38
  %24 = add i64 %arg.seed, %23
  %25 = and i64 %arg.iters, 3
  %26 = insertelement <4 x i64> <i64 poison, i64 0, i64 0, i64 0>, i64 %bc.merge.rdx, i64 0
  %broadcast.splatinsert9 = insertelement <4 x i64> poison, i64 %bc.resume.val, i64 0
  %broadcast.splat10 = shufflevector <4 x i64> %broadcast.splatinsert9, <4 x i64> poison, <4 x i32> zeroinitializer
  %induction11 = add <4 x i64> %broadcast.splat10, <i64 0, i64 -7046029254386353131, i64 4354685564936845354, i64 -2691343689449507777>
  %27 = sub i64 %vec.epilog.resume.val, %n.vec8
  br label %vec.epilog.vector.body

vec.epilog.vector.body:                           ; preds = %vec.epilog.vector.body, %vec.epilog.ph
  %lsr.iv34 = phi i64 [ %lsr.iv.next35, %vec.epilog.vector.body ], [ %27, %vec.epilog.ph ]
  %vec.phi13 = phi <4 x i64> [ %26, %vec.epilog.ph ], [ %31, %vec.epilog.vector.body ]
  %vec.ind14 = phi <4 x i64> [ %induction11, %vec.epilog.ph ], [ %vec.ind.next16, %vec.epilog.vector.body ]
  %28 = add <4 x i64> %vec.ind14, splat (i64 -7046029254386353131)
  %29 = lshr <4 x i64> %28, splat (i64 58)
  %30 = add <4 x i64> %vec.phi13, splat (i64 1)
  %31 = add <4 x i64> %30, %29
  %vec.ind.next16 = add <4 x i64> %vec.ind14, splat (i64 8709371129873690708)
  %lsr.iv.next35 = add i64 %lsr.iv34, 4
  %32 = icmp eq i64 %lsr.iv.next35, 0
  br i1 %32, label %vec.epilog.middle.block, label %vec.epilog.vector.body, !llvm.loop !4

vec.epilog.middle.block:                          ; preds = %vec.epilog.vector.body
  %rdx.shuf44 = shufflevector <4 x i64> %31, <4 x i64> poison, <4 x i32> <i32 2, i32 3, i32 poison, i32 poison>
  %bin.rdx45 = add <4 x i64> %31, %rdx.shuf44
  %rdx.shuf46 = shufflevector <4 x i64> %bin.rdx45, <4 x i64> poison, <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>
  %bin.rdx47 = add <4 x i64> %bin.rdx45, %rdx.shuf46
  %33 = extractelement <4 x i64> %bin.rdx47, i32 0
  %cmp.n17 = icmp eq i64 %arg.iters, %n.vec8
  br i1 %cmp.n17, label %B30.B144_crit_edge, label %B34.preheader

B34.preheader:                                    ; preds = %iter.check, %vec.epilog.iter.check, %vec.epilog.middle.block
  %.17984.ph = phi i64 [ %.175.promoted, %iter.check ], [ %20, %vec.epilog.iter.check ], [ %33, %vec.epilog.middle.block ]
  %x.2.081.ph = phi i64 [ %arg.seed, %iter.check ], [ %ind.end, %vec.epilog.iter.check ], [ %24, %vec.epilog.middle.block ]
  %.1067680.ph = phi i64 [ %arg.iters, %iter.check ], [ %ind.end21, %vec.epilog.iter.check ], [ %25, %vec.epilog.middle.block ]
  %34 = add i64 %x.2.081.ph, %const38
  %35 = add nuw nsw i64 %.1067680.ph, 1
  br label %B34

B34:                                              ; preds = %B34.preheader, %B34
  %lsr.iv32 = phi i64 [ %35, %B34.preheader ], [ %lsr.iv.next33, %B34 ]
  %lsr.iv = phi i64 [ %34, %B34.preheader ], [ %lsr.iv.next, %B34 ]
  %.17984 = phi i64 [ %.179, %B34 ], [ %.17984.ph, %B34.preheader ]
  %.177 = lshr i64 %lsr.iv, 58
  %.178 = add i64 %.17984, 1
  %.179 = add i64 %.178, %.177
  %lsr.iv.next = add i64 %lsr.iv, %const38
  %lsr.iv.next33 = add nsw i64 %lsr.iv32, -1
  %.107 = icmp samesign ugt i64 %lsr.iv.next33, 1
  br i1 %.107, label %B34, label %B30.B144_crit_edge, !llvm.loop !5

B30.B144_crit_edge:                               ; preds = %B34, %vec.epilog.middle.block, %middle.block
  %.179.lcssa = phi i64 [ %33, %vec.epilog.middle.block ], [ %20, %middle.block ], [ %.179, %B34 ]
  %.neg = mul i64 %arg.iters, %const38
  %36 = add i64 %.neg, %arg.seed
  %sunkaddr = mul i64 %arg.slot, 8
  %sunkaddr48 = getelementptr i8, ptr %arg.buffer.4, i64 %sunkaddr
  store i64 %.179.lcssa, ptr %sunkaddr48, align 8
  br label %B144

B144:                                             ; preds = %B30.B144_crit_edge, %B0.endif
  %x.2.0.lcssa = phi i64 [ %36, %B30.B144_crit_edge ], [ %arg.seed, %B0.endif ]
  store i64 %x.2.0.lcssa, ptr %retptr, align 8
  ret i32 0
}

define ptr @_ZN7cpython8__main__27naive_promotable_accumulateB2v2B40c8tJTIeFIjxB2IKSgI4CrvQClQZ6FcpCShpgEU0AE5ArrayIyLi1E1C7mutable7alignedEyyy(ptr readnone captures(none) %py_closure, ptr %py_args, ptr readnone captures(none) %py_kws) local_unnamed_addr {
entry:
  %.5 = alloca ptr, align 8
  %.6 = alloca ptr, align 8
  %.7 = alloca ptr, align 8
  %.8 = alloca ptr, align 8
  %.9 = call i32 (ptr, ptr, i64, i64, ...) @PyArg_UnpackTuple(ptr %py_args, ptr nonnull @.const.naive_promotable_accumulate, i64 4, i64 4, ptr nonnull %.5, ptr nonnull %.6, ptr nonnull %.7, ptr nonnull %.8)
  %.10 = icmp eq i32 %.9, 0
  %.22 = alloca { ptr, ptr, i64, i64, ptr, [1 x i64], [1 x i64] }, align 8
  %.93 = alloca i64, align 8
  store i64 0, ptr %.93, align 8
  br i1 %.10, label %common.ret, label %entry.endif, !prof !6

common.ret:                                       ; preds = %entry.endif.endif.endif.thread, %arg0.err, %entry, %entry.endif.endif.endif.e...endif.endif, %entry.endif.if
  %common.ret.op = phi ptr [ null, %arg0.err ], [ null, %entry ], [ null, %entry.endif.if ], [ null, %entry.endif.endif.endif.thread ], [ %.120, %entry.endif.endif.endif.e...endif.endif ]
  ret ptr %common.ret.op

entry.endif:                                      ; preds = %entry
  %.14 = load ptr, ptr @_ZN08NumbaEnv8__main__27naive_promotable_accumulateB2v2B40c8tJTIeFIjxB2IKSgI4CrvQClQZ6FcpCShpgEU0AE5ArrayIyLi1E1C7mutable7alignedEyyy, align 8
  %.17 = icmp eq ptr %.14, null
  br i1 %.17, label %entry.endif.if, label %entry.endif.endif, !prof !6

entry.endif.if:                                   ; preds = %entry.endif
  call void @PyErr_SetString(ptr nonnull @PyExc_RuntimeError, ptr nonnull @".const.missing Environment: _ZN08NumbaEnv8__main__27naive_promotable_accumulateB2v2B40c8tJTIeFIjxB2IKSgI4CrvQClQZ6FcpCShpgEU0AE5ArrayIyLi1E1C7mutable7alignedEyyy")
  br label %common.ret

entry.endif.endif:                                ; preds = %entry.endif
  %.21 = load ptr, ptr %.5, align 8
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(56) %.22, i8 0, i64 56, i1 false)
  %.26 = call i32 @NRT_adapt_ndarray_from_python(ptr %.21, ptr nonnull %.22)
  %sunkaddr = getelementptr inbounds i8, ptr %.22, i64 24
  %.30 = load i64, ptr %sunkaddr, align 8
  %.31 = icmp ne i64 %.30, 8
  %.32 = icmp ne i32 %.26, 0
  %.33 = or i1 %.32, %.31
  br i1 %.33, label %entry.endif.endif.endif.thread, label %entry.endif.endif.endif.endif, !prof !6

entry.endif.endif.endif.thread:                   ; preds = %entry.endif.endif
  call void @PyErr_SetString(ptr nonnull @PyExc_TypeError, ptr nonnull @".const.can't unbox array from PyObject into native value.  The object maybe of a different type")
  br label %common.ret

entry.endif.endif.endif.endif:                    ; preds = %entry.endif.endif
  %.37.fca.0.load = load ptr, ptr %.22, align 8
  %sunkaddr1 = getelementptr inbounds i8, ptr %.22, i64 32
  %.37.fca.4.load = load ptr, ptr %sunkaddr1, align 8
  %.44 = load ptr, ptr %.6, align 8
  %.47 = call ptr @PyNumber_Long(ptr %.44)
  %.48.not = icmp eq ptr %.47, null
  br i1 %.48.not, label %entry.endif.endif.endif.endif.endif, label %entry.endif.endif.endif.endif.if, !prof !6

arg0.err:                                         ; preds = %entry.endif.endif.endif.e...endif, %entry.endif.endif.endif.endif.endif.endif.endif, %entry.endif.endif.endif.endif.endif
  call void @NRT_decref(ptr %.37.fca.0.load)
  br label %common.ret

entry.endif.endif.endif.endif.if:                 ; preds = %entry.endif.endif.endif.endif
  %.50 = call i64 @PyLong_AsUnsignedLongLong(ptr nonnull %.47)
  call void @Py_DecRef(ptr nonnull %.47)
  br label %entry.endif.endif.endif.endif.endif

entry.endif.endif.endif.endif.endif:              ; preds = %entry.endif.endif.endif.endif.if, %entry.endif.endif.endif.endif
  %.45.0 = phi i64 [ %.50, %entry.endif.endif.endif.endif.if ], [ 0, %entry.endif.endif.endif.endif ]
  %.55 = call ptr @PyErr_Occurred()
  %.56.not = icmp eq ptr %.55, null
  br i1 %.56.not, label %entry.endif.endif.endif.endif.endif.endif, label %arg0.err, !prof !7

entry.endif.endif.endif.endif.endif.endif:        ; preds = %entry.endif.endif.endif.endif.endif
  %.60 = load ptr, ptr %.7, align 8
  %.63 = call ptr @PyNumber_Long(ptr %.60)
  %.64.not = icmp eq ptr %.63, null
  br i1 %.64.not, label %entry.endif.endif.endif.endif.endif.endif.endif, label %entry.endif.endif.endif.endif.endif.endif.if, !prof !6

entry.endif.endif.endif.endif.endif.endif.if:     ; preds = %entry.endif.endif.endif.endif.endif.endif
  %.66 = call i64 @PyLong_AsUnsignedLongLong(ptr nonnull %.63)
  call void @Py_DecRef(ptr nonnull %.63)
  br label %entry.endif.endif.endif.endif.endif.endif.endif

entry.endif.endif.endif.endif.endif.endif.endif:  ; preds = %entry.endif.endif.endif.endif.endif.endif.if, %entry.endif.endif.endif.endif.endif.endif
  %.61.0 = phi i64 [ %.66, %entry.endif.endif.endif.endif.endif.endif.if ], [ 0, %entry.endif.endif.endif.endif.endif.endif ]
  %.71 = call ptr @PyErr_Occurred()
  %.72.not = icmp eq ptr %.71, null
  br i1 %.72.not, label %entry.endif.endif.endif.endif.endif.endif.endif.endif, label %arg0.err, !prof !7

entry.endif.endif.endif.endif.endif.endif.endif.endif: ; preds = %entry.endif.endif.endif.endif.endif.endif.endif
  %.76 = load ptr, ptr %.8, align 8
  %.79 = call ptr @PyNumber_Long(ptr %.76)
  %.80.not = icmp eq ptr %.79, null
  br i1 %.80.not, label %entry.endif.endif.endif.e...endif, label %entry.endif.endif.endif.e...if, !prof !6

entry.endif.endif.endif.e...if:                   ; preds = %entry.endif.endif.endif.endif.endif.endif.endif.endif
  %.82 = call i64 @PyLong_AsUnsignedLongLong(ptr nonnull %.79)
  call void @Py_DecRef(ptr nonnull %.79)
  br label %entry.endif.endif.endif.e...endif

entry.endif.endif.endif.e...endif:                ; preds = %entry.endif.endif.endif.e...if, %entry.endif.endif.endif.endif.endif.endif.endif.endif
  %.77.0 = phi i64 [ %.82, %entry.endif.endif.endif.e...if ], [ 0, %entry.endif.endif.endif.endif.endif.endif.endif.endif ]
  %.87 = call ptr @PyErr_Occurred()
  %.88.not = icmp eq ptr %.87, null
  br i1 %.88.not, label %entry.endif.endif.endif.e...endif.endif, label %arg0.err, !prof !7

entry.endif.endif.endif.e...endif.endif:          ; preds = %entry.endif.endif.endif.e...endif
  %.92 = call ptr @PyEval_SaveThread()
  store i64 0, ptr %.93, align 8
  %.99 = call i32 @_ZN8__main__27naive_promotable_accumulateB2v2B40c8tJTIeFIjxB2IKSgI4CrvQClQZ6FcpCShpgEU0AE5ArrayIyLi1E1C7mutable7alignedEyyy(ptr nonnull %.93, ptr nonnull poison, ptr poison, ptr poison, i64 poison, i64 poison, ptr %.37.fca.4.load, i64 poison, i64 poison, i64 %.45.0, i64 %.61.0, i64 %.77.0) #3
  %.109 = load i64, ptr %.93, align 8
  call void @PyEval_RestoreThread(ptr %.92)
  call void @NRT_decref(ptr %.37.fca.0.load)
  %.120 = call ptr @PyLong_FromUnsignedLongLong(i64 %.109)
  br label %common.ret
}

declare i32 @PyArg_UnpackTuple(ptr, ptr, i64, i64, ...) local_unnamed_addr

declare void @PyErr_SetString(ptr, ptr) local_unnamed_addr

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #1

declare i32 @NRT_adapt_ndarray_from_python(ptr captures(none), ptr captures(none)) local_unnamed_addr

declare ptr @PyNumber_Long(ptr) local_unnamed_addr

declare i64 @PyLong_AsUnsignedLongLong(ptr) local_unnamed_addr

declare void @Py_DecRef(ptr) local_unnamed_addr

declare ptr @PyErr_Occurred() local_unnamed_addr

declare ptr @PyEval_SaveThread() local_unnamed_addr

declare void @PyEval_RestoreThread(ptr) local_unnamed_addr

declare ptr @PyLong_FromUnsignedLongLong(i64) local_unnamed_addr

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none, target_mem0: none, target_mem1: none)
define i64 @cfunc._ZN8__main__27naive_promotable_accumulateB2v2B40c8tJTIeFIjxB2IKSgI4CrvQClQZ6FcpCShpgEU0AE5ArrayIyLi1E1C7mutable7alignedEyyy({ ptr, ptr, i64, i64, ptr, [1 x i64], [1 x i64] } %.1, i64 %.2, i64 %.3, i64 %.4) local_unnamed_addr #2 {
entry:
  %.6 = alloca i64, align 8
  store i64 0, ptr %.6, align 8
  %extracted.data = extractvalue { ptr, ptr, i64, i64, ptr, [1 x i64], [1 x i64] } %.1, 4
  %.12 = call i32 @_ZN8__main__27naive_promotable_accumulateB2v2B40c8tJTIeFIjxB2IKSgI4CrvQClQZ6FcpCShpgEU0AE5ArrayIyLi1E1C7mutable7alignedEyyy(ptr nonnull %.6, ptr nonnull poison, ptr poison, ptr poison, i64 poison, i64 poison, ptr %extracted.data, i64 poison, i64 poison, i64 %.2, i64 %.3, i64 %.4) #3
  %.22 = load i64, ptr %.6, align 8
  ret i64 %.22
}

; Function Attrs: noinline
define linkonce_odr void @NRT_decref(ptr %.1) local_unnamed_addr #3 {
.3:
  %.4 = icmp eq ptr %.1, null
  br i1 %.4, label %common.ret1, label %.3.endif, !prof !6

common.ret1:                                      ; preds = %.3, %.3.endif
  ret void

.3.endif:                                         ; preds = %.3
  fence release
  %0 = tail call i8 @llvm.x86.atomic.sub.cc.i64(ptr nonnull %.1, i64 1, i32 4)
  %1 = trunc i8 %0 to i1
  br i1 %1, label %.3.endif.if, label %common.ret1, !prof !6

.3.endif.if:                                      ; preds = %.3.endif
  fence acquire
  tail call void @NRT_MemInfo_call_dtor(ptr nonnull %.1)
  ret void
}

; Function Attrs: nounwind
declare i8 @llvm.x86.atomic.sub.cc.i64(ptr, i64, i32 immarg) #4

declare void @NRT_MemInfo_call_dtor(ptr) local_unnamed_addr

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.vector.reduce.add.v8i64(<8 x i64>) #5

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.vector.reduce.add.v4i64(<4 x i64>) #5

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) }
attributes #1 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #2 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none, target_mem0: none, target_mem1: none) }
attributes #3 = { noinline }
attributes #4 = { nounwind }
attributes #5 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!0 = distinct !{!0, !1, !2}
!1 = !{!"llvm.loop.isvectorized", i32 1}
!2 = !{!"llvm.loop.unroll.runtime.disable"}
!3 = !{!"branch_weights", i32 4, i32 28}
!4 = distinct !{!4, !1, !2}
!5 = distinct !{!5, !2, !1}
!6 = !{!"branch_weights", i32 1, i32 99}
!7 = !{!"branch_weights", i32 99, i32 1}
