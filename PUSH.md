# دفع كود التوقعات إلى GitHub (المستودع الصحيح)

المستودع المصدر للحظة: الرمز محضّر ومُلتزم محلياً هنا (للنسخ الذي يخصّ لا ليغا/ليغ 1).

الهدف على GitHub: `https://github.com/JinTheHuman/FootBallSportsBettingAid` (بفرع `main`)

## لماذا لم يُدفع تلقائياً
الجهاز الحالي يتعرف بـ `75nn12345872168` (حساب مستودع الدواجن)، وليس لديه صلاحية دفع إلى
مستودع `JinTheHuman`. لم أستطع ولن أتعامل مع كلمة مرور/رمز حساب آخر.

## الدفع (أمر واحد، بعد تسجيل دخول حساب JinTheHuman)
	# تأكد من أن git يتعرف بحساب JinTheHuman (مثلاً عبر `git push` من جهاز/اعتماد ذلك الحساب)، ثم:
	git -C "<مسار المشروع>/Downloads/mom66/FootBallSportsBettingAid" push origin main

## بديل (لو كان التوقعات يجب أن تكون تحت حساب 75nn12345872168)
	# أنشئ مستودعك بهذا الحساب ثم بدّل الـ remote وادفع:
	git remote set-url origin https://github.com/75nn12345872168/<اسم-المستودع>.git
	git push -u origin main

## بعد الدفع (لتشغيل جدول Telegram من المستودع الصحيح)
1. في مستودع التوقعات على GitHub → Settings → Secrets → Actions وأضف:
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
   - `THE_ODDS_API_KEY` (اختياري — لتفعيل +EV مؤكد بالأسعار الحقيقية)
2. فعّل الـ workflow التفريج في تبويب Actions، أو انتظر الجدول (إثنين 09:00 / جمعة 18:00 UTC).
3. عطّل الـ workflow القديم في مستودع الدواجن حتى لا يتضاعف الإرسال.

## تنظيف مستودع الدواجن (اختياري، غير-مدمر)
بعد تأكيد دفع التوقعات للمستودع الصحيح:
	git -C "<مسار>/poultry" rm -- .github/workflows/autoloop.yml autoloop_master.py settle_ledger_auto.py fetch_odds.py …
	git commit -m "remove wrongly-pushed betting pipeline"
	git push origin main
(أنا جاهز لتنفيذه لحظة تأكيدك؛ إعادة كتابة التاريخ / force-push لن أنفذها دون أمر صريح.)