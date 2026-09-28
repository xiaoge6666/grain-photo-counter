package com.example.ricecounter;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import androidx.core.content.FileProvider;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.InputStream;

public class MainActivity extends Activity {
    private static final int PICK_IMAGE = 1;
    private static final int TAKE_PHOTO = 2;
    private TextView resultText;
    private ImageView resultImage;
    private Button zoomBtn;
    private Bitmap resultBitmap;
    private Uri photoUri;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        ScrollView scroll = new ScrollView(this);
        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        layout.setPadding(50, 80, 50, 50);

        TextView title = new TextView(this);
        title.setTextSize(22);
        title.setText("稻谷计数（ASW-SPC 算法）");
        layout.addView(title);

        Button photoBtn = new Button(this);
        photoBtn.setText("拍照计数");
        photoBtn.setTextSize(18);
        photoBtn.setLayoutParams(margins(40));
        photoBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                takePhoto();
            }
        });
        layout.addView(photoBtn);

        Button pickBtn = new Button(this);
        pickBtn.setText("从相册选图计数");
        pickBtn.setTextSize(18);
        pickBtn.setLayoutParams(margins(20));
        pickBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                startActivityForResult(Intent.createChooser(intent, "选择图片"), PICK_IMAGE);
            }
        });
        layout.addView(pickBtn);

        resultText = new TextView(this);
        resultText.setTextSize(22);
        resultText.setText("点上方按钮开始");
        resultText.setPadding(0, 40, 0, 10);
        layout.addView(resultText);

        resultImage = new ImageView(this);
        resultImage.setAdjustViewBounds(true);
        resultImage.setLayoutParams(new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT));
        resultImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) { showZoom(); }
        });
        layout.addView(resultImage);

        zoomBtn = new Button(this);
        zoomBtn.setText("🔍 放大查看结果图（双指缩放/拖动/双击）");
        zoomBtn.setTextSize(16);
        zoomBtn.setEnabled(false);
        zoomBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) { showZoom(); }
        });
        layout.addView(zoomBtn);

        TextView hint = new TextView(this);
        hint.setTextSize(13);
        hint.setText("👆 点结果图或上方按钮可全屏放大，逐粒核对编号");
        hint.setPadding(0, 8, 0, 16);
        layout.addView(hint);

        scroll.addView(layout);
        setContentView(scroll);
    }

    private LinearLayout.LayoutParams margins(int top) {
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        lp.topMargin = top;
        return lp;
    }

    private void takePhoto() {
        File dir = new File(getCacheDir(), "images");
        dir.mkdirs();
        File photoFile = new File(dir, "photo.jpg");
        photoUri = FileProvider.getUriForFile(this,
                getApplicationContext().getPackageName() + ".fileprovider", photoFile);
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        intent.putExtra(MediaStore.EXTRA_OUTPUT, photoUri);
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        startActivityForResult(intent, TAKE_PHOTO);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_OK) return;

        final byte[] bytes;
        if (requestCode == TAKE_PHOTO) {
            bytes = readBytes(photoUri);
        } else if (requestCode == PICK_IMAGE) {
            if (data == null || data.getData() == null) return;
            bytes = readBytes(data.getData());
        } else {
            return;
        }

        if (bytes == null) {
            resultText.setText("读取图片失败");
            return;
        }

        resultText.setText("正在计数...");
        new Thread(new Runnable() {
            @Override
            public void run() {
                final int[] count = new int[1];
                final byte[][] annotated = new byte[1][];
                try {
                    Python py = Python.getInstance();
                    PyObject module = py.getModule("main");
                    PyObject result = module.callAttr("count_and_label", bytes);
                    count[0] = result.asList().get(0).toInt();
                    annotated[0] = result.asList().get(1).toJava(byte[].class);
                } catch (Exception e) {
                    count[0] = -1;
                }
                final int c = count[0];
                final byte[] ab = annotated[0];
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        if (c < 0) {
                            resultText.setText("计数失败（错误码 " + c + "）");
                            resultImage.setImageBitmap(null);
                            zoomBtn.setEnabled(false);
                        } else {
                            resultText.setText("计数结果: " + c + " 粒");
                            if (ab != null && ab.length > 0) {
                                resultBitmap = BitmapFactory.decodeByteArray(ab, 0, ab.length);
                                resultImage.setImageBitmap(resultBitmap);
                                zoomBtn.setEnabled(true);
                            }
                        }
                    }
                });
            }
        }).start();
    }

    /** 全屏放大查看结果图 */
    private void showZoom() {
        if (resultBitmap == null || resultBitmap.isRecycled()) return;
        try {
            final android.app.Dialog d = new android.app.Dialog(this,
                    android.R.style.Theme_Black_NoTitleBar_Fullscreen);
            LinearLayout ll = new LinearLayout(this);
            ll.setOrientation(LinearLayout.VERTICAL);
            ZoomImageView zv = new ZoomImageView(this, resultBitmap);
            ll.addView(zv, new LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f));
            TextView hint = new TextView(this);
            hint.setTextSize(14);
            hint.setTextColor(0xFFFFFFFF);
            hint.setText("双指缩放 · 单指拖动 · 双击放大/还原");
            hint.setPadding(24, 16, 24, 8);
            ll.addView(hint);
            Button close = new Button(this);
            close.setText("关闭");
            close.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) { d.dismiss(); }
            });
            ll.addView(close);
            d.setContentView(ll);
            d.show();
        } catch (Throwable e) {
            resultText.setText("放大失败: " + e.getMessage());
        }
    }

    private byte[] readBytes(Uri uri) {
        try {
            InputStream is = getContentResolver().openInputStream(uri);
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            byte[] buffer = new byte[8192];
            int n;
            while ((n = is.read(buffer)) != -1) {
                baos.write(buffer, 0, n);
            }
            is.close();
            return baos.toByteArray();
        } catch (Exception e) {
            return null;
        }
    }
}
