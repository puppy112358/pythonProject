import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
file_path = '附件1：123家有信贷记录企业的相关数据.xlsx'
data = pd.read_excel(file_path, None)

# 提取每个工作表的数据
company_info = data['企业信息']
input_invoices = data['进项发票信息']
output_invoices = data['销项发票信息']

# 计算每个企业的进项和销项总金额
input_summary = input_invoices.groupby('企业代号')['价税合计'].sum().reset_index()
input_summary.columns = ['企业代号', '进项总金额']

output_summary = output_invoices.groupby('企业代号')['价税合计'].sum().reset_index()
output_summary.columns = ['企业代号', '销项总金额']

# 合并数据
company_data = company_info.merge(input_summary, on='企业代号', how='left').merge(output_summary, on='企业代号', how='left')
company_data['进项总金额'].fillna(0, inplace=True)
company_data['销项总金额'].fillna(0, inplace=True)

# 编码违约情况
company_data['违约'] = company_data['是否违约'].map({'是': 1, '否': 0})

# 选择特征和目标变量
features = company_data[['进项总金额', '销项总金额']]
target = company_data['违约']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# 创建随机森林模型
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 预测和评估模型
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy}")
print(classification_report(y_test, y_pred))

# 提取特征重要性
feature_importances = rf_model.feature_importances_
importance_df = pd.DataFrame({'Feature': features.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# 输出特征重要性
print("特征重要性:")
print(importance_df)

# 可视化特征重要性
plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance in Random Forest Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
