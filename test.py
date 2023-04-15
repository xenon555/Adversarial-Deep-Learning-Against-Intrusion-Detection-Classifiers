import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from cleverhans.attacks_tf import jacobian_graph, jsma, fgsm
from cleverhans.utils_tf import model_train, model_eval, batch_eval
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow.python.platform import flags

#import matplotlib.pyplot as plt
plt.style.use('bmh')
#将图表样式设置为“bmh”。

FLAGS = flags.FLAGS
flags.DEFINE_integer('nb_epochs', 20, 'Number of epochs to train model')
#nb_epochs'代表训练模型的epoch数
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
#'batch_size'代表训练批次的大小
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')
#'learning_rate'代表训练的学习率
flags.DEFINE_integer('nb_classes', 5, 'Number of classification classes')
#'nb_classes'代表分类的数量
flags.DEFINE_integer('source_samples', 10, 'Nb of test set examples to attack')
#'source_samples'代表攻击测试集的样本数
print()
print()
print(" -------------- Start of preprocessing stage --- ---- ----- --")

names = ['duration', 'protocol ', 'service ', 'flag ', 'src_bytes ', 'dst_bytes ', 'land ',
         'wrong_fragment ',
         'urgent ', 'hot ', 'num_failed_logins ', 'logged_in ', ' num_compromised ',
         'root_shell ', ' su_attempted ',
         'num_root ', ' num_file_creations ', 'num_shells ', ' num_access_files ',
         'num_outbound_cmds ',
         'is_host_login ', 'is_guest_login ', 'count ', 'srv_count ', ' serror_rate ',
         'srv_serror_rate ',
         'rerror_rate ', 'srv_rerror_rate ', 'same_srv_rate ', 'diff_srv_rate ',
         'srv_diff_host_rate ',
         'dst_host_count ', ' dst_host_srv_count ', ' dst_host_same_srv_rate ',
         'dst_host_diff_srv_rate ',
         'dst_host_same_src_port_rate ', ' dst_host_srv_diff_host_rate ',
         'dst_host_serror_rate ',
         'dst_host_srv_serror_rate ', ' dst_host_rerror_rate ', 'dst_host_srv_rerror_rate ',
         'attack_type ', 'other ']

df = pd.read_csv('kddcup.data_10_percent_corrected ', names=names, header=None)
dft = pd.read_csv('KDDTest +. txt ', names=names, header=None)
# 'KDDTrain12 +. txt '，'KDDTest +. txt '
print(" Initial test and training data shapes :", df.shape, dft.shape)

full = pd.concat([df, dft])
# 将两个数据集拼接成一个数据集full
assert full.shape[0] == df.shape[0] + dft.shape[0]

full['label'] = full['attack_type']
# 将原始数据集的标签列 'attack_type' 进行修改，将不同类型的攻击归为以下四类：
# 修改后的标签保存在新的列 'label' 中，同时删除了原始标签列 'attack_type' 和 'other'。
# DoS attacks：       （Denial of Service，拒绝服务）
full.loc[full.label == 'neptune', 'label'] = 'dos '
full.loc[full.label == 'back ', 'label'] = 'dos'
full.loc[full.label == 'land ', 'label '] = 'dos '
full.loc[full.label == 'pod ', 'label '] = 'dos '
full.loc[full.label == 'smurf ', 'label '] = 'dos '
full.loc[full.label == 'teardrop ', 'label '] = 'dos '
full.loc[full.label == 'mailbomb ', 'label '] = 'dos '
full.loc[full.label == ' processtable ', 'label '] = 'dos '
full.loc[full.label == 'udpstorm ', 'label '] = 'dos '
full.loc[full.label == 'apache2 ', 'label '] = 'dos '
full.loc[full.label == 'worm ', 'label '] = 'dos '

# User -to - Root ( U2R )       （User-to-Root，用户到根）
full.loc[full.label == ' buffer_overflow ', 'label '] = 'u2r '
full.loc[full.label == 'loadmodule ', 'label '] = 'u2r '
full.loc[full.label == 'perl ', 'label '] = 'u2r '
full.loc[full.label == 'rootkit ', 'label '] = 'u2r '
full.loc[full.label == 'sqlattack ', 'label '] = 'u2r '
full.loc[full.label == 'xterm ', 'label '] = 'u2r '
full.loc[full.label == 'ps ', 'label '] = 'u2r '

# Remote -to - Local ( R2L)      （Remote-to-Local，远程到本地）
full.loc[full.label == 'ftp_write ', 'label '] = 'r2l '
full.loc[full.label == ' guess_passwd ', 'label '] = 'r2l '
full.loc[full.label == 'imap ', 'label '] = 'r2l '
full.loc[full.label == 'multihop ', 'label '] = 'r2l '
full.loc[full.label == 'phf ', 'label '] = 'r2l '
full.loc[full.label == 'spy ', 'label '] = 'r2l '
full.loc[full.label == ' warezclient ', 'label '] = 'r2l '
full.loc[full.label == ' warezmaster ', 'label '] = 'r2l '
full.loc[full.label == 'xlock ', 'label '] = 'r2l '
full.loc[full.label == 'xsnoop ', 'label '] = 'r2l '
full.loc[full.label == ' snmpgetattack ', 'label '] = 'r2l '
full.loc[full.label == 'httptunnel ', 'label '] = 'r2l '
full.loc[full.label == 'snmpguess ', 'label '] = 'r2l '
full.loc[full.label == 'sendmail ', 'label '] = 'r2l '
full.loc[full.label == 'named ', 'label '] = 'r2l '

# Probe attacls       探测攻击
full.loc[full.label == 'satan ', 'label '] = 'probe '
full.loc[full.label == 'ipsweep ', 'label '] = 'probe '
full.loc[full.label == 'nmap ', 'label '] = 'probe '
full.loc[full.label == 'portsweep ', 'label '] = 'probe '
full.loc[full.label == 'saint ', 'label '] = 'probe '
full.loc[full.label == 'mscan ', 'label '] = 'probe '

full = full.drop(['other ', 'attack_type '], axis=1)
# 从full数据集中删除’other‘ 和 ’attack_type‘两列，drop()函数用于删除指定的行或列，'axis=1'表示删除列
print("Unique labels ", full.label.unique())

# Generate One -Hot encoding
full2 = pd.get_dummies(full, drop_first=False)

# Separate training and test sets again
features = list(full2.columns[: -5])
y_train = np.array(full2[0: df.shape[0]][[' label_normal ', 'label_dos ', ' label_probe ', 'label_r2l ', 'label_u2r ']])
X_train = full2[0: df.shape[0]][features]
y_test = np.array(full2[df.shape[0]:][[' label_normal ', 'label_dos ', 'label_probe ', 'label_r2l ', 'label_u2r ']])
X_test = full2[df.shape[0]:][features]

# Scale data
scaler = MinMaxScaler().fit(X_train)
X_train_scaled = np.array(scaler.transform(X_train))
X_test_scaled = np.array(scaler.transform(X_test))

# Generate label encoding for Logistic regression
labels = full.label.unique()
le = LabelEncoder()
le.fit(labels)
y_full = le.transform(full.label)
y_train_l = y_full[0: df.shape[0]]
y_test_l = y_full[df.shape[0]:]

print(" Training dataset shape ", X_train_scaled.shape, y_train.shape)
print(" Test dataset shape ", X_test_scaled.shape, y_test.shape)
print(" Label encoder y shape ", y_train_l.shape, y_test_l.shape)

print(" --------------End of preprocessing stage ----- ---- -----")
print()
print()

print(" -------------- Start of adversarial sample generation - ---- ----- ----")
print()


def mlp_model():
    """
    Generate a MultiLayer Perceptron model
    """
    model = Sequential()
    model.add(Dense(256, activation='relu ', input_shape=(X_train_scaled.shape[1],)))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation='relu '))
    model.add(Dropout(0.4))
    model.add(Dense(FLAGS.nb_classes, activation='softmax '))
    model.compile(loss=' categorical_crossentropy ', optimizer='adam ', metrics=['accuracy '])

    model.summary()
    return model


def evaluate():
    """
    Model evaluation function
    """
    eval_params = {'batch_size ': FLAGS.batch_size}
    accuracy = model_eval(sess, x, y, predictions, X_test_scaled, y_test, args=eval_params)
    print('Test accuracy on legitimate test examples : ' + str(accuracy))


# Tensorflow placeholder variables
x = tf.placeholder(tf.float32, shape=(None, X_train_scaled.shape[1]))
y = tf.placeholder(tf.float32, shape=(None, FLAGS.nb_classes))

tf.set_random_seed(42)
model = mlp_model()
sess = tf.Session()
predictions = model(x)
init = tf.global_variables_initializer()
sess.run(init)
# Train the model
train_params = {
    'nb_epochs ': FLAGS.nb_epochs,
    'batch_size ': FLAGS.batch_size,
    ' learning_rate ': FLAGS.learning_rate,
    'verbose ': 0
}
model_train(sess, x, y, predictions, X_train_scaled, y_train, evaluate=evaluate, args=
train_params)

# Generate adversarial samples for all test datapoints
source_samples = X_test_scaled.shape[0]

# Jacobian - based Saliency Map
results = np.zeros((FLAGS.nb_classes, source_samples), dtype='i')
perturbations = np.zeros((FLAGS.nb_classes, source_samples), dtype='f')
grads = jacobian_graph(predictions, x, FLAGS.nb_classes)

X_adv = np.zeros((source_samples, X_test_scaled.shape[1]))

for sample_ind in range(0, source_samples):
    # We want to find an adversarial example for each possible target class
    # (i.e. all classes that differ from the label given in the dataset )
    current_class = int(np.argmax(y_test[sample_ind]))
    # target_classes = other_classes ( FLAGS . nb_classes , current_class )

    # Only target the normal class
    for target in [0]:
        # print ( ' - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ')
        # print (' Creating adv . example for target class ' + str ( target ))
        if current_class == 0:
            break
        # This call runs the Jacobian - based saliency map approach
        adv_x, res, percent_perturb = jsma(sess, x, predictions, grads,
                                           X_test_scaled[sample_ind: (sample_ind + 1)],
                                           target, theta=1, gamma=0.1,
                                           increase=True, back='tf ',
                                           clip_min=0, clip_max=1)

        X_adv[sample_ind] = adv_x
        results[target, sample_ind] = res
        perturbations[target, sample_ind] = percent_perturb

print(X_adv.shape)

print(" -------------- Evaluation of MLP performance -- ---- ----- ---")
print()
eval_params = {'batch_size ': FLAGS.batch_size}
accuracy = model_eval(sess, x, y, predictions, X_test_scaled, y_test,
                      args=eval_params)
print('Test accuracy on normal examples : ' + str(accuracy))

accuracy_adv = model_eval(sess, x, y, predictions, X_adv, y_test,
                          args=eval_params)
print('Test accuracy on adversarial examples : ' + str(accuracy_adv))

print()
print(" ------------ Decision Tree Classifier - ---- ----- -")
dt = OneVsRestClassifier(DecisionTreeClassifier(random_state=42))
dt.fit(X_train_scaled, y_train)
y_pred = dt.predict(X_test_scaled)

# Calculate FPR for normal class only
fpr_dt, tpr_dt, _ = roc_curve(y_test[:, 0], y_pred[:, 0])

roc_auc_dt = auc(fpr_dt, tpr_dt)
print(" Accuracy score :", accuracy_score(y_test, y_pred))
print("F1 score :", f1_score(y_test, y_pred, average='micro '))
print(" AUC score :", roc_auc_dt)

# Predict using adversarial test samples
y_pred_adv = dt.predict(X_adv)
fpr_dt_adv, tpr_dt_adv, _ = roc_curve(y_test[:, 0], y_pred_adv[:, 0])
roc_auc_dt_adv = auc(fpr_dt_adv, tpr_dt_adv)
print(" Accuracy score adversarial :", accuracy_score(y_test, y_pred_adv))
print("F1 score adversarial :", f1_score(y_test, y_pred_adv, average='micro'))
print(" AUC score adversarial :", roc_auc_dt_adv)

plt.figure()
lw = 2
plt.plot(fpr_dt, tpr_dt, color='darkorange ',
         lw=lw, label='ROC curve ( area = %0.2 f)' % roc_auc_dt)
plt.plot(fpr_dt_adv, tpr_dt_adv, color='green ',
         lw=lw, label='ROC curve adv .( area = %0.2 f)' % roc_auc_dt_adv)
plt.plot([0, 1], [0, 1], color='navy ', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate ')
plt.ylabel('True Positive Rate ')
plt.title('ROC Decision Tree ( class = Normal )')
plt.legend(loc=" lower right ")
plt.savefig('ROC_DT . png ')

print()
print()
print(" ------------ Random Forest Classifier - ---- ----- -")
rf = OneVsRestClassifier(RandomForestClassifier(n_estimators=200, random_state=42))
rf.fit(X_train_scaled, y_train)
y_pred = rf.predict(X_test_scaled)

fpr_rf, tpr_rf, _ = roc_curve(y_test[:, 0], y_pred[:, 0])
roc_auc_rf = auc(fpr_rf, tpr_rf)
print(" Accuracy score :", accuracy_score(y_test, y_pred))
print("F1 score :", f1_score(y_test, y_pred, average='micro '))
print(" AUC score :", roc_auc_rf)

# Predict using adversarial test samples
y_pred_adv = rf.predict(X_adv)
fpr_rf_adv, tpr_rf_adv, _ = roc_curve(y_test[:, 0], y_pred_adv[:, 0])
roc_auc_rf_adv = auc(fpr_rf_adv, tpr_rf_adv)
print(" Accuracy score adversarial :", accuracy_score(y_test, y_pred_adv))
print("F1 score adversarial :", f1_score(y_test, y_pred_adv, average='micro '))
print(" AUC score adversarial :", roc_auc_rf_adv)

plt.figure()
lw = 2
plt.plot(fpr_rf, tpr_rf, color='darkorange ',
         lw=lw, label='ROC curve ( area = %0.2 f)' % roc_auc_rf)
plt.plot(fpr_rf_adv, tpr_rf_adv, color='green ',
         lw=lw, label='ROC curve adv . ( area = %0.2 f)' % roc_auc_rf_adv)
plt.plot([0, 1], [0, 1], color='navy ', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate ')
plt.ylabel('True Positive Rate ')
plt.title('ROC Random Forest ( class = Normal )')
plt.legend(loc=" lower right ")
plt.savefig('ROC_RF . png ')

print()
print()
print(" ------------ Linear SVM Classifier ---- ---- ---")
sv = OneVsRestClassifier(LinearSVC(C=1., random_state=42, loss='hinge '))
sv.fit(X_train_scaled, y_train)

y_pred = sv.predict(X_test_scaled)
fpr_sv, tpr_sv, _ = roc_curve(y_test[:, 0], y_pred[:, 0])
roc_auc_sv = auc(fpr_sv, tpr_sv)
print(" Accuracy score :", accuracy_score(y_test, y_pred))
print("F1 score :", f1_score(y_test, y_pred, average='micro'))
print(" AUC score :", roc_auc_sv)

# Predict using adversarial test samples
y_pred_adv = sv.predict(X_adv)
fpr_sv_adv, tpr_sv_adv, _ = roc_curve(y_test[:, 0], y_pred_adv[:, 0])
roc_auc_sv_adv = auc(fpr_sv_adv, tpr_sv_adv)
print(" Accuracy score adversarial ", accuracy_score(y_test, y_pred_adv))
print("F1 score adversarial :", f1_score(y_test, y_pred_adv, average='micro '))
print(" AUC score adversarial :", roc_auc_sv_adv)

plt.figure()
lw = 2
plt.plot(fpr_sv, tpr_sv, color='darkorange ',
         lw=lw, label='ROC curve ( area = %0.2 f)' % roc_auc_sv)
plt.plot(fpr_sv_adv, tpr_sv_adv, color='green ',
         lw=lw, label='ROC curve adv .( area = %0.2 f)' % roc_auc_sv_adv)
plt.plot([0, 1], [0, 1], color='navy ', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate ')
plt.ylabel('True Positive Rate ')
plt.title('ROC SVM ( class = Normal )')
plt.legend(loc=" lower right ")
plt.savefig('ROC_SVM .png ')

print()
print()
print(" ------------ Voting Classifier ---- ----- --")
vot = VotingClassifier(estimators=[('dt ', dt), ('rf ', rf), ('sv ', sv)], voting='hard ')
vot.fit(X_train_scaled, y_train_l)

y_pred = vot.predict(X_test_scaled)
fpr_vot, tpr_vot, _ = roc_curve(y_test_l, y_pred, pos_label=1, drop_intermediate=False
                                )
roc_auc_vot = auc(fpr_vot, tpr_vot)
print(" Accuracy score ", accuracy_score(y_test_l, y_pred))
print("F1 score :", f1_score(y_test_l, y_pred, average='micro '))
print(" AUC score :", roc_auc_vot)

# Predict using adversarial test samples
y_pred_adv = vot.predict(X_adv)
fpr_vot_adv, tpr_vot_adv, _ = roc_curve(y_test_l, y_pred_adv, pos_label=1,
                                        drop_intermediate=False)
roc_auc_vot_adv = auc(fpr_vot_adv, tpr_vot_adv)
print(" Accuracy score adversarial :", accuracy_score(y_test_l, y_pred_adv))
print("F1 score adversarial :", f1_score(y_test_l, y_pred_adv, average='micro '))
print(" AUC score adversarial :", roc_auc_vot_adv)

plt.figure()
lw = 2
plt.plot(fpr_vot, tpr_vot, color='darkorange ',
         lw=lw, label='ROC curve ( area = %0.2 f)' % roc_auc_vot)
plt.plot(fpr_vot_adv, tpr_vot_adv, color='green ',
         lw=lw, label='ROC curve adv . ( area = %0.2 f)' % roc_auc_vot_adv)
plt.plot([0, 1], [0, 1], color='navy ', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate ')
plt.ylabel('True Positive Rate ')
plt.title('ROC Voting ( class = Normal )')
plt.legend(loc=" lower right ")
plt.savefig('ROC_Vot .png ')

# Print overall ROC curves
plt.figure(figsize=(12, 6))
plt.plot(fpr_dt_adv, tpr_dt_adv, label='DT ( area = %0.2 f)' % roc_auc_dt_adv)
plt.plot(fpr_rf_adv, tpr_rf_adv, label='RF ( area = %0.2 f)' % roc_auc_rf_adv)
plt.plot(fpr_sv_adv, tpr_sv_adv, label='SVM ( area = %0.2 f)' % roc_auc_sv_adv)
plt.plot(fpr_vot_adv, tpr_vot_adv, label='Vot ( area = %0.2 f)' % roc_auc_vot_adv)
# plt . plot ( fpr_lr_adv , tpr_lr_adv , label =’LR ( area = %0.2 f)’ % roc_auc_lr_adv )

plt.xlabel('False positive rate ')
plt.ylabel('True positive rate ')
plt.title('ROC curve ( adversarial samples )')
plt.legend(loc='best ')
plt.savefig(' ROC_curves_adv . png ')

plt.figure(figsize=(12, 6))
plt.plot(fpr_dt, tpr_dt, label='DT ( area = %0.2 f)' % roc_auc_dt)
plt.plot(fpr_rf, tpr_rf, label='RF ( area = %0.2 f)' % roc_auc_rf)
plt.plot(fpr_sv, tpr_sv, label='SVM ( area = %0.2 f)' % roc_auc_sv)
plt.plot(fpr_vot, tpr_vot, label='Vot ( area = %0.2 f)' % roc_auc_vot)

# plt . plot ( fpr_lr , tpr_lr , label ='LR ( area = %0.2 f)' % roc_auc_lr )

plt.xlabel('False positive rate ')
plt.ylabel('True positive rate ')
plt.title('ROC curve ( normal samples )')
plt.legend(loc='best ')
plt.savefig('ROC_curves . png ')

print()
print()
print(" ------------ Adversarial feature statistics ---- ----- --")
feats = dict()
total = 0
orig_attack = X_test_scaled - X_adv
for i in range(0, orig_attack.shape[0]):

    ind = np.where(orig_attack[i, :] != 0)[0]
    total += len(ind)
    for j in ind:
        if j in feats:
            feats[j] += 1
        else:
            feats[j] = 1

# The number of features that where changed for the adversarial samples
print(" Number of unique features changed :", len(feats.keys()))
print(" Number of average features changed per datapoint ", total / len(orig_attack))

top_10 = sorted(feats, key=feats.get, reverse=True)[:10]
top_20 = sorted(feats, key=feats.get, reverse=True)[:20]
print(" Top ten features :", X_test.columns[top_10])

top_10_val = [100 * feats[k] / y_test.shape[0] for k in top_10]
top_20_val = [100 * feats[k] / y_test.shape[0] for k in top_20]

plt.figure(figsize=(16, 12))
plt.bar(np.arange(20), top_20_val, align='center ')
plt.xticks(np.arange(20), X_test.columns[top_20], rotation='vertical ')
plt.title('Feature participation in adversarial examples ')
plt.ylabel('Percentage (%) ')
plt.xlabel('Features ')
plt.savefig(' Adv_features . png ')

# Craft adversarial examples using Fast Gradient Sign Method ( FGSM )
adv_x_f = fgsm(x, predictions, eps=0.3)
X_test_adv, = batch_eval(sess, [x], [adv_x_f], [X_test_scaled])

# Evaluate the accuracy of the MNIST model on adversarial examples
accuracy = model_eval(sess, x, y, predictions, X_test_adv, y_test)
print('Test accuracy on adversarial examples : ' + str(accuracy))

# Comparison of adversarial and original test samples ( attack )
feats = dict()
total = 0
orig_attack = X_test_scaled - X_test_adv
for i in range(0, orig_attack.shape[0]):

    ind = np.where(orig_attack[i, :] != 0)[0]
    total += len(ind)
    for j in ind:
        if j in feats:
            feats[j] += 1
        else:
            feats[j] = 1

# The number of features that were changed for the adversarial samples
print(" Number of unique features changed with FGSM :", len(feats.keys()))
print(" Number of average features changed per datapoint with FGSM :", total / len(
    orig_attack))
