; ModuleID = 'worker_strided_accumulate'
source_filename = "<string>"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

@.const.worker_strided_accumulate = internal constant [26 x i8] c"worker_strided_accumulate\00"
@_ZN08NumbaEnv8__main__25worker_strided_accumulateB2v1B40c8tJTIeFIjxB2IKSgI4CrvQClQZ6FcpCShpgEU0AE5ArrayIyLi1E1C7mutable7alignedEyyyy = common local_unnamed_addr global ptr null
@".const.missing Environment: _ZN08NumbaEnv8__main__25worker_strided_accumulateB2v1B40c8tJTIeFIjxB2IKSgI4CrvQClQZ6FcpCShpgEU0AE5ArrayIyLi1E1C7mutable7alignedEyyyy" = internal constant [154 x i8] c"missing Environment: _ZN08NumbaEnv8__main__25worker_strided_accumulateB2v1B40c8tJTIeFIjxB2IKSgI4CrvQClQZ6FcpCShpgEU0AE5ArrayIyLi1E1C7mutable7alignedEyyyy\00"
@PyExc_TypeError = external global i8
@".const.can't unbox array from PyObject into native value.  The object maybe of a different type" = internal constant [89 x i8] c"can't unbox array from PyObject into native value.  The object maybe of a different type\00"
@PyExc_RuntimeError = external global i8

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define noundef i32 @_ZN8__main__25worker_strided_accumulateB2v1B40c8tJTIeFIjxB2IKSgI4CrvQClQZ6FcpCShpgEU0AE5ArrayIyLi1E1C7mutable7alignedEyyyy(ptr noalias writeonly captures(none) %retptr, ptr noalias readnone captures(none) %excinfo, ptr readnone captures(none) %arg.buffer.0, ptr readnone captures(none) %arg.buffer.1, i64 %arg.buffer.2, i64 %arg.buffer.3, ptr captures(none) %arg.buffer.4, i64 %arg.buffer.5.0, i64 %arg.buffer.6.0, i64 %arg.slot, i64 %arg.iters, i64 %arg.seed, i64 %arg.mask) local_unnamed_addr #0 {
B0.endif:
  %.11078.not = icmp slt i64 %arg.iters, 1
  br i1 %.11078.not, label %B174, label %B34.lr.ph

B34.lr.ph:                                        ; preds = %B0.endif
  %const = bitcast i64 -7046029254386353131 to i64
  %0 = getelementptr i64, ptr %arg.buffer.4, i64 %arg.slot
  %xtraiter = and i64 %arg.iters, 7
  %lcmp.mod.not = icmp eq i64 %xtraiter, 0
  br i1 %lcmp.mod.not, label %B34.prol.loopexit, label %B34.prol.preheader

B34.prol.preheader:                               ; preds = %B34.lr.ph
  br label %B34.prol

B34.prol:                                         ; preds = %B34.prol.preheader, %B34.prol
  %x.2.081.prol = phi i64 [ %.149.prol, %B34.prol ], [ %arg.seed, %B34.prol.preheader ]
  %prol.iter = phi i64 [ %prol.iter.next, %B34.prol ], [ 0, %B34.prol.preheader ]
  %.149.prol = add i64 %x.2.081.prol, %const
  %.153.prol = and i64 %.149.prol, %arg.mask
  %.181.prol = getelementptr i64, ptr %0, i64 %.153.prol
  %.182.prol = load i64, ptr %.181.prol, align 8
  %.183.prol = lshr i64 %.149.prol, 58
  %.184.prol = add nuw nsw i64 %.183.prol, 1
  %.185.prol = add i64 %.184.prol, %.182.prol
  store i64 %.185.prol, ptr %.181.prol, align 8
  %prol.iter.next = add i64 %prol.iter, 1
  %prol.iter.cmp.not = icmp eq i64 %xtraiter, %prol.iter.next
  br i1 %prol.iter.cmp.not, label %B34.prol.loopexit.loopexit, label %B34.prol, !llvm.loop !0

B34.prol.loopexit.loopexit:                       ; preds = %B34.prol
  %1 = sub i64 %arg.iters, %prol.iter.next
  br label %B34.prol.loopexit

B34.prol.loopexit:                                ; preds = %B34.prol.loopexit.loopexit, %B34.lr.ph
  %x.2.081.unr = phi i64 [ %arg.seed, %B34.lr.ph ], [ %.149.prol, %B34.prol.loopexit.loopexit ]
  %.1097680.unr = phi i64 [ %arg.iters, %B34.lr.ph ], [ %1, %B34.prol.loopexit.loopexit ]
  %.149.lcssa.unr = phi i64 [ poison, %B34.lr.ph ], [ %.149.prol, %B34.prol.loopexit.loopexit ]
  %2 = icmp ult i64 %arg.iters, 8
  br i1 %2, label %B174, label %B34.preheader

B34.preheader:                                    ; preds = %B34.prol.loopexit
  %3 = add i64 %.1097680.unr, 8
  br label %B34

B34:                                              ; preds = %B34.preheader, %B34
  %lsr.iv = phi i64 [ %3, %B34.preheader ], [ %lsr.iv.next, %B34 ]
  %x.2.081 = phi i64 [ %.149.7, %B34 ], [ %x.2.081.unr, %B34.preheader ]
  %.149 = add i64 %x.2.081, %const
  %.153 = and i64 %.149, %arg.mask
  %.181 = getelementptr i64, ptr %0, i64 %.153
  %.182 = load i64, ptr %.181, align 8
  %.183 = lshr i64 %.149, 58
  %.184 = add nuw nsw i64 %.183, 1
  %.185 = add i64 %.184, %.182
  store i64 %.185, ptr %.181, align 8
  %.149.1 = add i64 %x.2.081, 4354685564936845354
  %.153.1 = and i64 %.149.1, %arg.mask
  %.181.1 = getelementptr i64, ptr %0, i64 %.153.1
  %.182.1 = load i64, ptr %.181.1, align 8
  %.183.1 = lshr i64 %.149.1, 58
  %.184.1 = add nuw nsw i64 %.183.1, 1
  %.185.1 = add i64 %.184.1, %.182.1
  store i64 %.185.1, ptr %.181.1, align 8
  %.149.2 = add i64 %x.2.081, -2691343689449507777
  %.153.2 = and i64 %.149.2, %arg.mask
  %.181.2 = getelementptr i64, ptr %0, i64 %.153.2
  %.182.2 = load i64, ptr %.181.2, align 8
  %.183.2 = lshr i64 %.149.2, 58
  %.184.2 = add nuw nsw i64 %.183.2, 1
  %.185.2 = add i64 %.184.2, %.182.2
  store i64 %.185.2, ptr %.181.2, align 8
  %.149.3 = add i64 %x.2.081, 8709371129873690708
  %.153.3 = and i64 %.149.3, %arg.mask
  %.181.3 = getelementptr i64, ptr %0, i64 %.153.3
  %.182.3 = load i64, ptr %.181.3, align 8
  %.183.3 = lshr i64 %.149.3, 58
  %.184.3 = add nuw nsw i64 %.183.3, 1
  %.185.3 = add i64 %.184.3, %.182.3
  store i64 %.185.3, ptr %.181.3, align 8
  %.149.4 = add i64 %x.2.081, 1663341875487337577
  %.153.4 = and i64 %.149.4, %arg.mask
  %.181.4 = getelementptr i64, ptr %0, i64 %.153.4
  %.182.4 = load i64, ptr %.181.4, align 8
  %.183.4 = lshr i64 %.149.4, 58
  %.184.4 = add nuw nsw i64 %.183.4, 1
  %.185.4 = add i64 %.184.4, %.182.4
  store i64 %.185.4, ptr %.181.4, align 8
  %.149.5 = add i64 %x.2.081, -5382687378899015554
  %.153.5 = and i64 %.149.5, %arg.mask
  %.181.5 = getelementptr i64, ptr %0, i64 %.153.5
  %.182.5 = load i64, ptr %.181.5, align 8
  %.183.5 = lshr i64 %.149.5, 58
  %.184.5 = add nuw nsw i64 %.183.5, 1
  %.185.5 = add i64 %.184.5, %.182.5
  store i64 %.185.5, ptr %.181.5, align 8
  %.149.6 = add i64 %x.2.081, 6018027440424182931
  %.153.6 = and i64 %.149.6, %arg.mask
  %.181.6 = getelementptr i64, ptr %0, i64 %.153.6
  %.182.6 = load i64, ptr %.181.6, align 8
  %.183.6 = lshr i64 %.149.6, 58
  %.184.6 = add nuw nsw i64 %.183.6, 1
  %.185.6 = add i64 %.184.6, %.182.6
  store i64 %.185.6, ptr %.181.6, align 8
  %.149.7 = add i64 %x.2.081, -1028001813962170200
  %.153.7 = and i64 %.149.7, %arg.mask
  %.181.7 = getelementptr i64, ptr %0, i64 %.153.7
  %.182.7 = load i64, ptr %.181.7, align 8
  %.183.7 = lshr i64 %.149.7, 58
  %.184.7 = add nuw nsw i64 %.183.7, 1
  %.185.7 = add i64 %.184.7, %.182.7
  store i64 %.185.7, ptr %.181.7, align 8
  %lsr.iv.next = add i64 %lsr.iv, -8
  %.110.7 = icmp sgt i64 %lsr.iv.next, 8
  br i1 %.110.7, label %B34, label %B174

B174:                                             ; preds = %B34, %B34.prol.loopexit, %B0.endif
  %x.2.0.lcssa = phi i64 [ %arg.seed, %B0.endif ], [ %.149.lcssa.unr, %B34.prol.loopexit ], [ %.149.7, %B34 ]
  store i64 %x.2.0.lcssa, ptr %retptr, align 8
  ret i32 0
}

define ptr @_ZN7cpython8__main__25worker_strided_accumulateB2v1B40c8tJTIeFIjxB2IKSgI4CrvQClQZ6FcpCShpgEU0AE5ArrayIyLi1E1C7mutable7alignedEyyyy(ptr readnone captures(none) %py_closure, ptr %py_args, ptr readnone captures(none) %py_kws) local_unnamed_addr {
entry:
  %.5 = alloca ptr, align 8
  %.6 = alloca ptr, align 8
  %.7 = alloca ptr, align 8
  %.8 = alloca ptr, align 8
  %.9 = alloca ptr, align 8
  %.10 = call i32 (ptr, ptr, i64, i64, ...) @PyArg_UnpackTuple(ptr %py_args, ptr nonnull @.const.worker_strided_accumulate, i64 5, i64 5, ptr nonnull %.5, ptr nonnull %.6, ptr nonnull %.7, ptr nonnull %.8, ptr nonnull %.9)
  %.11 = icmp eq i32 %.10, 0
  %.23 = alloca { ptr, ptr, i64, i64, ptr, [1 x i64], [1 x i64] }, align 8
  %.110 = alloca i64, align 8
  store i64 0, ptr %.110, align 8
  br i1 %.11, label %common.ret, label %entry.endif, !prof !2

common.ret:                                       ; preds = %entry.endif.endif.endif.thread, %arg0.err, %entry, %entry.endif.endif.endif.e...endif.endif.endif.endif, %entry.endif.if
  %common.ret.op = phi ptr [ null, %arg0.err ], [ null, %entry ], [ null, %entry.endif.if ], [ null, %entry.endif.endif.endif.thread ], [ %.137, %entry.endif.endif.endif.e...endif.endif.endif.endif ]
  ret ptr %common.ret.op

entry.endif:                                      ; preds = %entry
  %.15 = load ptr, ptr @_ZN08NumbaEnv8__main__25worker_strided_accumulateB2v1B40c8tJTIeFIjxB2IKSgI4CrvQClQZ6FcpCShpgEU0AE5ArrayIyLi1E1C7mutable7alignedEyyyy, align 8
  %.18 = icmp eq ptr %.15, null
  br i1 %.18, label %entry.endif.if, label %entry.endif.endif, !prof !2

entry.endif.if:                                   ; preds = %entry.endif
  call void @PyErr_SetString(ptr nonnull @PyExc_RuntimeError, ptr nonnull @".const.missing Environment: _ZN08NumbaEnv8__main__25worker_strided_accumulateB2v1B40c8tJTIeFIjxB2IKSgI4CrvQClQZ6FcpCShpgEU0AE5ArrayIyLi1E1C7mutable7alignedEyyyy")
  br label %common.ret

entry.endif.endif:                                ; preds = %entry.endif
  %.22 = load ptr, ptr %.5, align 8
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(56) %.23, i8 0, i64 56, i1 false)
  %.27 = call i32 @NRT_adapt_ndarray_from_python(ptr %.22, ptr nonnull %.23)
  %sunkaddr = getelementptr inbounds i8, ptr %.23, i64 24
  %.31 = load i64, ptr %sunkaddr, align 8
  %.32 = icmp ne i64 %.31, 8
  %.33 = icmp ne i32 %.27, 0
  %.34 = or i1 %.33, %.32
  br i1 %.34, label %entry.endif.endif.endif.thread, label %entry.endif.endif.endif.endif, !prof !2

entry.endif.endif.endif.thread:                   ; preds = %entry.endif.endif
  call void @PyErr_SetString(ptr nonnull @PyExc_TypeError, ptr nonnull @".const.can't unbox array from PyObject into native value.  The object maybe of a different type")
  br label %common.ret

entry.endif.endif.endif.endif:                    ; preds = %entry.endif.endif
  %.38.fca.0.load = load ptr, ptr %.23, align 8
  %sunkaddr1 = getelementptr inbounds i8, ptr %.23, i64 32
  %.38.fca.4.load = load ptr, ptr %sunkaddr1, align 8
  %.45 = load ptr, ptr %.6, align 8
  %.48 = call ptr @PyNumber_Long(ptr %.45)
  %.49.not = icmp eq ptr %.48, null
  br i1 %.49.not, label %entry.endif.endif.endif.endif.endif, label %entry.endif.endif.endif.endif.if, !prof !2

arg0.err:                                         ; preds = %entry.endif.endif.endif.e...endif.endif.endif, %entry.endif.endif.endif.e...endif, %entry.endif.endif.endif.endif.endif.endif.endif, %entry.endif.endif.endif.endif.endif
  call void @NRT_decref(ptr %.38.fca.0.load)
  br label %common.ret

entry.endif.endif.endif.endif.if:                 ; preds = %entry.endif.endif.endif.endif
  %.51 = call i64 @PyLong_AsUnsignedLongLong(ptr nonnull %.48)
  call void @Py_DecRef(ptr nonnull %.48)
  br label %entry.endif.endif.endif.endif.endif

entry.endif.endif.endif.endif.endif:              ; preds = %entry.endif.endif.endif.endif.if, %entry.endif.endif.endif.endif
  %.46.0 = phi i64 [ %.51, %entry.endif.endif.endif.endif.if ], [ 0, %entry.endif.endif.endif.endif ]
  %.56 = call ptr @PyErr_Occurred()
  %.57.not = icmp eq ptr %.56, null
  br i1 %.57.not, label %entry.endif.endif.endif.endif.endif.endif, label %arg0.err, !prof !3

entry.endif.endif.endif.endif.endif.endif:        ; preds = %entry.endif.endif.endif.endif.endif
  %.61 = load ptr, ptr %.7, align 8
  %.64 = call ptr @PyNumber_Long(ptr %.61)
  %.65.not = icmp eq ptr %.64, null
  br i1 %.65.not, label %entry.endif.endif.endif.endif.endif.endif.endif, label %entry.endif.endif.endif.endif.endif.endif.if, !prof !2

entry.endif.endif.endif.endif.endif.endif.if:     ; preds = %entry.endif.endif.endif.endif.endif.endif
  %.67 = call i64 @PyLong_AsUnsignedLongLong(ptr nonnull %.64)
  call void @Py_DecRef(ptr nonnull %.64)
  br label %entry.endif.endif.endif.endif.endif.endif.endif

entry.endif.endif.endif.endif.endif.endif.endif:  ; preds = %entry.endif.endif.endif.endif.endif.endif.if, %entry.endif.endif.endif.endif.endif.endif
  %.62.0 = phi i64 [ %.67, %entry.endif.endif.endif.endif.endif.endif.if ], [ 0, %entry.endif.endif.endif.endif.endif.endif ]
  %.72 = call ptr @PyErr_Occurred()
  %.73.not = icmp eq ptr %.72, null
  br i1 %.73.not, label %entry.endif.endif.endif.endif.endif.endif.endif.endif, label %arg0.err, !prof !3

entry.endif.endif.endif.endif.endif.endif.endif.endif: ; preds = %entry.endif.endif.endif.endif.endif.endif.endif
  %.77 = load ptr, ptr %.8, align 8
  %.80 = call ptr @PyNumber_Long(ptr %.77)
  %.81.not = icmp eq ptr %.80, null
  br i1 %.81.not, label %entry.endif.endif.endif.e...endif, label %entry.endif.endif.endif.e...if, !prof !2

entry.endif.endif.endif.e...if:                   ; preds = %entry.endif.endif.endif.endif.endif.endif.endif.endif
  %.83 = call i64 @PyLong_AsUnsignedLongLong(ptr nonnull %.80)
  call void @Py_DecRef(ptr nonnull %.80)
  br label %entry.endif.endif.endif.e...endif

entry.endif.endif.endif.e...endif:                ; preds = %entry.endif.endif.endif.e...if, %entry.endif.endif.endif.endif.endif.endif.endif.endif
  %.78.0 = phi i64 [ %.83, %entry.endif.endif.endif.e...if ], [ 0, %entry.endif.endif.endif.endif.endif.endif.endif.endif ]
  %.88 = call ptr @PyErr_Occurred()
  %.89.not = icmp eq ptr %.88, null
  br i1 %.89.not, label %entry.endif.endif.endif.e...endif.endif, label %arg0.err, !prof !3

entry.endif.endif.endif.e...endif.endif:          ; preds = %entry.endif.endif.endif.e...endif
  %.93 = load ptr, ptr %.9, align 8
  %.96 = call ptr @PyNumber_Long(ptr %.93)
  %.97.not = icmp eq ptr %.96, null
  br i1 %.97.not, label %entry.endif.endif.endif.e...endif.endif.endif, label %entry.endif.endif.endif.e...endif.endif.if, !prof !2

entry.endif.endif.endif.e...endif.endif.if:       ; preds = %entry.endif.endif.endif.e...endif.endif
  %.99 = call i64 @PyLong_AsUnsignedLongLong(ptr nonnull %.96)
  call void @Py_DecRef(ptr nonnull %.96)
  br label %entry.endif.endif.endif.e...endif.endif.endif

entry.endif.endif.endif.e...endif.endif.endif:    ; preds = %entry.endif.endif.endif.e...endif.endif.if, %entry.endif.endif.endif.e...endif.endif
  %.94.0 = phi i64 [ %.99, %entry.endif.endif.endif.e...endif.endif.if ], [ 0, %entry.endif.endif.endif.e...endif.endif ]
  %.104 = call ptr @PyErr_Occurred()
  %.105.not = icmp eq ptr %.104, null
  br i1 %.105.not, label %entry.endif.endif.endif.e...endif.endif.endif.endif, label %arg0.err, !prof !3

entry.endif.endif.endif.e...endif.endif.endif.endif: ; preds = %entry.endif.endif.endif.e...endif.endif.endif
  %.109 = call ptr @PyEval_SaveThread()
  store i64 0, ptr %.110, align 8
  %.116 = call i32 @_ZN8__main__25worker_strided_accumulateB2v1B40c8tJTIeFIjxB2IKSgI4CrvQClQZ6FcpCShpgEU0AE5ArrayIyLi1E1C7mutable7alignedEyyyy(ptr nonnull %.110, ptr nonnull poison, ptr poison, ptr poison, i64 poison, i64 poison, ptr %.38.fca.4.load, i64 poison, i64 poison, i64 %.46.0, i64 %.62.0, i64 %.78.0, i64 %.94.0) #3
  %.126 = load i64, ptr %.110, align 8
  call void @PyEval_RestoreThread(ptr %.109)
  call void @NRT_decref(ptr %.38.fca.0.load)
  %.137 = call ptr @PyLong_FromUnsignedLongLong(i64 %.126)
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
define i64 @cfunc._ZN8__main__25worker_strided_accumulateB2v1B40c8tJTIeFIjxB2IKSgI4CrvQClQZ6FcpCShpgEU0AE5ArrayIyLi1E1C7mutable7alignedEyyyy({ ptr, ptr, i64, i64, ptr, [1 x i64], [1 x i64] } %.1, i64 %.2, i64 %.3, i64 %.4, i64 %.5) local_unnamed_addr #2 {
entry:
  %.7 = alloca i64, align 8
  store i64 0, ptr %.7, align 8
  %extracted.data = extractvalue { ptr, ptr, i64, i64, ptr, [1 x i64], [1 x i64] } %.1, 4
  %.13 = call i32 @_ZN8__main__25worker_strided_accumulateB2v1B40c8tJTIeFIjxB2IKSgI4CrvQClQZ6FcpCShpgEU0AE5ArrayIyLi1E1C7mutable7alignedEyyyy(ptr nonnull %.7, ptr nonnull poison, ptr poison, ptr poison, i64 poison, i64 poison, ptr %extracted.data, i64 poison, i64 poison, i64 %.2, i64 %.3, i64 %.4, i64 %.5) #3
  %.23 = load i64, ptr %.7, align 8
  ret i64 %.23
}

; Function Attrs: noinline
define linkonce_odr void @NRT_decref(ptr %.1) local_unnamed_addr #3 {
.3:
  %.4 = icmp eq ptr %.1, null
  br i1 %.4, label %common.ret1, label %.3.endif, !prof !2

common.ret1:                                      ; preds = %.3, %.3.endif
  ret void

.3.endif:                                         ; preds = %.3
  fence release
  %0 = tail call i8 @llvm.x86.atomic.sub.cc.i64(ptr nonnull %.1, i64 1, i32 4)
  %1 = trunc i8 %0 to i1
  br i1 %1, label %.3.endif.if, label %common.ret1, !prof !2

.3.endif.if:                                      ; preds = %.3.endif
  fence acquire
  tail call void @NRT_MemInfo_call_dtor(ptr nonnull %.1)
  ret void
}

; Function Attrs: nounwind
declare i8 @llvm.x86.atomic.sub.cc.i64(ptr, i64, i32 immarg) #4

declare void @NRT_MemInfo_call_dtor(ptr) local_unnamed_addr

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) }
attributes #1 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #2 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none, target_mem0: none, target_mem1: none) }
attributes #3 = { noinline }
attributes #4 = { nounwind }

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
!2 = !{!"branch_weights", i32 1, i32 99}
!3 = !{!"branch_weights", i32 99, i32 1}
