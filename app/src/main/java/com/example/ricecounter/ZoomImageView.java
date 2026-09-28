package com.example.ricecounter;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.view.GestureDetector;
import android.view.MotionEvent;
import android.view.ScaleGestureDetector;
import android.view.View;

/** 可缩放/拖动的图片视图：双指缩放、单指拖动、双击放大/还原 */
public class ZoomImageView extends View {
    private final Bitmap bmp;
    private final Matrix fit = new Matrix();
    private final Matrix user = new Matrix();
    private final Matrix total = new Matrix();
    private float fitScale = 1f;
    private float scale = 1f;
    private final float maxFactor = 10f;
    private float lastX, lastY;
    private boolean dragging = false;
    private final ScaleGestureDetector sgd;
    private final GestureDetector gd;
    private final Paint paint = new Paint(Paint.FILTER_BITMAP_FLAG | Paint.ANTI_ALIAS_FLAG);

    public ZoomImageView(Context c, Bitmap b) {
        super(c);
        bmp = b;
        sgd = new ScaleGestureDetector(c, new ScaleGestureDetector.SimpleOnScaleGestureListener() {
            @Override
            public boolean onScale(ScaleGestureDetector d) {
                float f = d.getScaleFactor();
                float ns = scale * f;
                if (ns < fitScale) f = fitScale / scale;
                if (ns > fitScale * maxFactor) f = fitScale * maxFactor / scale;
                scale *= f;
                user.postScale(f, f, d.getFocusX(), d.getFocusY());
                invalidate();
                return true;
            }
        });
        gd = new GestureDetector(c, new GestureDetector.SimpleOnGestureListener() {
            @Override
            public boolean onDown(MotionEvent e) { return true; }

            @Override
            public boolean onDoubleTap(MotionEvent e) {
                float target = (scale > fitScale * 1.5f) ? fitScale : fitScale * 4f;
                float f = target / scale;
                scale = target;
                user.postScale(f, f, e.getX(), e.getY());
                invalidate();
                return true;
            }
        });
    }

    @Override
    protected void onSizeChanged(int w, int h, int ow, int oh) {
        super.onSizeChanged(w, h, ow, oh);
        if (w == 0 || h == 0 || bmp == null) return;
        fit.reset();
        float s = Math.min(w / (float) bmp.getWidth(), h / (float) bmp.getHeight());
        fitScale = s;
        fit.postScale(s, s);
        fit.postTranslate((w - bmp.getWidth() * s) / 2f, (h - bmp.getHeight() * s) / 2f);
        user.reset();
        scale = s;
    }

    @Override
    protected void onDraw(Canvas cv) {
        if (bmp == null || bmp.isRecycled()) return;
        total.set(fit);
        total.postConcat(user);
        cv.drawBitmap(bmp, total, paint);
    }

    @Override
    public boolean onTouchEvent(MotionEvent e) {
        sgd.onTouchEvent(e);
        gd.onTouchEvent(e);
        switch (e.getActionMasked()) {
            case MotionEvent.ACTION_DOWN:
                lastX = e.getX(); lastY = e.getY(); dragging = true;
                break;
            case MotionEvent.ACTION_POINTER_DOWN:
                dragging = false;
                break;
            case MotionEvent.ACTION_MOVE:
                if (dragging && !sgd.isInProgress() && e.getPointerCount() == 1) {
                    user.postTranslate(e.getX() - lastX, e.getY() - lastY);
                    lastX = e.getX(); lastY = e.getY();
                    invalidate();
                }
                break;
            case MotionEvent.ACTION_UP:
            case MotionEvent.ACTION_CANCEL:
                dragging = false;
                break;
        }
        return true;
    }
}
