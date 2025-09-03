# --- Component 2: Advanced Reinforcement Learning Model with Comprehensive Features --- 
import numpy as np
import random
import sys
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import KFold

try:
    trained_classifier_model = trained_classifier_model
    selected_malicious_log = malicious_log_for_rl
    kmeans_model = kmeans_model_for_rl
    if selected_malicious_log is None or not hasattr(trained_classifier_model, 'predict_proba') or not hasattr(kmeans_model, 'predict'):
        raise ValueError("One or more required components are not available.")
except (NameError, ValueError) as e:
    print(f"Warning: Could not access classifier model or malicious log from Component 1. {e}. Using placeholders.")
    
if selected_malicious_log is None or not hasattr(trained_classifier_model, 'predict_proba') or not hasattr(kmeans_model, 'predict'):
    print("Cannot run without a malicious log and models. Exiting.")
    sys.exit()
try:
    N_FEATURES_LOG = kmeans_model.n_features_in_
    N_FEATURES_CLASSIFIER = trained_classifier_model.n_features_in_
except AttributeError:
    print("Warning: Models do not have n_features_in_. Using fallback values.")
    N_FEATURES_LOG = selected_malicious_log.shape[0]
    N_FEATURES_CLASSIFIER = N_FEATURES_LOG + 1
print("--- Component 2: Advanced Reinforcement Learning Model ---")
def pad_log_to_features(log, target_features):
    """Pads a log array with zeros to match a target feature size."""
    if log.shape[0] < target_features:
        padding = target_features - log.shape[0]
        return np.pad(log, (0, padding), 'constant')
    return log[:target_features]
class AdvancedLinuxEnvironment:
    def __init__(self, initial_log, classifier, kmeans_model):
        self.classifier = classifier
        self.kmeans_model = kmeans_model
        self.log = pad_log_to_features(initial_log.copy(), N_FEATURES_LOG)
        self.state = self._get_state()
        self.deception_active = False
        self.is_malicious = True
    def _get_state(self):
        padded_log = pad_log_to_features(self.log.copy(), N_FEATURES_LOG)
        log_reshaped = padded_log.reshape(1, -1)
        cluster_prediction = self.kmeans_model.predict(log_reshaped)[0]
        mean_val = np.mean(self.log)
        if mean_val > 80:
            return f"high_activity_cluster_{cluster_prediction}"
        elif mean_val > 30:
            return f"medium_activity_cluster_{cluster_prediction}"
        else:
            return f"low_activity_cluster_{cluster_prediction}"
    def reset(self, initial_log):
        self.log = pad_log_to_features(initial_log.copy(), N_FEATURES_LOG)
        self.state = self._get_state()
        self.deception_active = False
        self.is_malicious = True
        return self.state
    def step(self, action_script):
        new_log = self.log.copy()
        action_reward = 0
        done = False
        if "rm" in action_script:
            new_log = np.zeros_like(new_log)
            action_reward = 150
            done = True
        elif "chown" in action_script:
            new_log = np.array([10]*len(new_log))
            action_reward = 120
            done = True
        elif "chmod" in action_script:
            new_log = np.array([20]*len(new_log))
            action_reward = 110
            done = True
        elif "deceive_ports" in action_script:
            new_log = np.random.randint(5, 20, size=len(new_log))
            action_reward = 90
            self.deception_active = True
            done = True
        elif "deceive_fs" in action_script:
            new_log = np.random.randint(10, 30, size=len(new_log))
            action_reward = 80
            self.deception_active = True
            done = True
        elif "deceive_service" in action_script:
            new_log = np.random.randint(15, 35, size=len(new_log))
            action_reward = 70
            self.deception_active = True
            done = True
        elif "echo" in action_script:
            new_log = np.random.randint(20, 40, size=len(new_log))
            action_reward = 40
            self.deception_active = True
            done = True
        elif any(cmd in action_script for cmd in ["systemctl", "firewall", "selinux", "kill", "ps", "netstat", "lsof", "auditctl", "fail2ban"]):
            new_log = np.array([10]*len(new_log))
            action_reward = 125
            done = True
        elif any(cmd in action_script for cmd in ["deceive_process", "fake_log", "deceive_user", "deceive_network", "deceive_config"]):
            new_log = np.random.randint(20, 40, size=len(new_log))
            action_reward = 85
            self.deception_active = True
            done = True
        else:
            action_reward = -100
            self.deception_active = False
        new_log = pad_log_to_features(new_log, N_FEATURES_LOG)
        log_reshaped = new_log.reshape(1, -1)
        cluster_prediction = self.kmeans_model.predict(log_reshaped)
        log_with_cluster = np.concatenate((log_reshaped, cluster_prediction.reshape(-1, 1)), axis=1)
        log_with_cluster = pad_log_to_features(log_with_cluster[0], N_FEATURES_CLASSIFIER).reshape(1, -1)
        classification_proba = self.classifier.predict_proba(log_with_cluster)[0, 1]
        self.is_malicious = (classification_proba > 0.5)
        if not self.is_malicious:
            classification_reward = 50
        else:
            classification_reward = -50
        total_reward = action_reward + classification_reward
        self.log = new_log
        new_state = self._get_state()
        return new_state, total_reward, done
class AdvancedNLPAgent:
    def __init__(self, actions):
        self.q_table = defaultdict(lambda: np.zeros(len(actions)))
        self.actions = actions
    def choose_action(self, state, exploration_rate, policy_type="epsilon-greedy"):
        if policy_type == "epsilon-greedy":
            if random.random() < exploration_rate:
                return random.randint(0, len(self.actions) - 1)
            else:
                return np.argmax(self.q_table[state])
        elif policy_type == "softmax":
            q_values = self.q_table[state]
            exp_q_values = np.exp(q_values - np.max(q_values))
            probabilities = exp_q_values / np.sum(exp_q_values)
            return np.random.choice(range(len(self.actions)), p=probabilities)
        else:
            raise ValueError("Invalid policy_type. Must be 'epsilon-greedy' or 'softmax'.")
    def update_policy(self, state, action_idx, reward, next_state, alpha, gamma):
        old_value = self.q_table[state][action_idx]
        next_max = np.max(self.q_table[next_state])
        self.q_table[state][action_idx] = old_value + alpha * (reward + gamma * next_max - old_value)
ACTION_SCRIPTS = [
    # --- Corrective Actions (50 scripts) ---
    "chown root:root /etc/shadow", "chmod 600 /etc/ssh/sshd_config", "rm -f /tmp/malicious.sh",
    "systemctl restart ssh", "iptables -A INPUT -p tcp --dport 22 -j DROP", "kill -9 $(pidof suspected_process)",
    "userdel -r compromised_user", "usermod -L compromised_user", "fail2ban-client ban --all",
    "auditctl -a always,exit -F arch=b64 -S open -F exit=-EPERM -k unauthorized_file_access",
    "semanage fcontext -a -t httpd_sys_content_t '/var/www(/.*)?'", "setsebool -P httpd_can_network_connect on",
    "netstat -tuln | grep 8080 | awk '{print $7}' | cut -d'/' -f1 | xargs kill",
    "lsof -iTCP -sTCP:LISTEN -P", "find / -name '*malware*' -delete", "clamscan -r / --move=/quarantine",
    "ufw reset", "ufw enable", "ufw deny incoming", "ufw allow 22/tcp",
    "rsync -avz /backup/files/ /", "iptables -P FORWARD DROP", "iptables -F",
    "echo 1 > /proc/sys/net/ipv4/tcp_syncookies", "sysctl -p", "route del default gw 192.168.1.1",
    "crontab -r", "find /tmp -type f -mtime +1 -delete", "truncate -s 0 /var/log/auth.log",
    "restorecon -R /var/www/html", "sed -i 's/^PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config",
    "grep 'failed password' /var/log/auth.log | awk '{print $11}' | sort | uniq -c | sort -nr",
    "chattr +i /etc/passwd", "chattr +a /var/log/audit", "e2fsck -f /dev/sda1",
    "fsck -y /dev/sda1", "fuser -k 8080/tcp", "killall -9 suspicious_process",
    "usermod -s /sbin/nologin compromised_user", "passwd -d compromised_user", "groupdel malicious_group",
    "rm -rf /root/.ssh/authorized_keys", "service sshd restart", "service httpd stop",
    "systemctl stop compromised_service", "systemctl disable compromised_service", "yum check-update",
    "apt-get update", "apt-get upgrade -y", "yum update -y",
    "dpkg -P malicious_package", "yum erase malicious_package", "find / -perm /6000",
    "find /var/www -name '*.php' -type f -exec grep -l -E 'eval\\(|base64_decode' {} \\;",
    "grep -r 'suspicious_ip' /var/log/apache2/", "find / -type f -name '.*' -ls",
    "ip route del 10.0.0.0/8", "iptables -A INPUT -s 1.2.3.4 -j DROP",
    # --- Deceptive Actions (50 scripts) ---
    "deceive_ports", "deceive_fs", "deceive_service", "echo 'System clean' > /var/log/fake_scan.log",
    "deceive_process_list", "deceive_network_traffic", "deceive_user_accounts", "deceive_file_permissions",
    "deceive_running_services", "deceive_configuration_files", "deceive_log_history",
    "touch -a -m -t 201010101010.10 /tmp/.malicious.sh", "iptables -A INPUT -j ACCEPT", "iptables -A OUTPUT -j ACCEPT",
    "service fake_service start", "mkdir /var/log/old_logs", "mv /var/log/auth.log /var/log/old_logs/auth.log.bak",
    "echo 'localhost' > /etc/hosts", "echo '127.0.0.1 www.example.com' >> /etc/hosts",
    "ln -s /dev/null /var/log/syslog", "echo 'malicious_user ALL=(ALL) NOPASSWD: ALL' > /etc/sudoers.d/fake_sudo",
    "cp /bin/ls /tmp/ls", "chmod u+s /tmp/ls", "cp /etc/passwd /tmp/passwd",
    "chmod 777 /var/www/html/", "chown nobody:nobody /var/www/html/",
    "echo 'HTTP/1.1 200 OK' > /var/www/html/index.php", "echo 'root:x:0:0:root:/root:/bin/bash' > /etc/shadow",
    "touch /var/tmp/.ssh_dir", "touch -t 200501010101.01 /etc/passwd", "touch /dev/null",
    "ln -s /dev/null /var/log/secure", "echo 'This is a test log.' > /var/log/secure",
    "mkdir -p /home/legit_user/.ssh", "touch /home/legit_user/.ssh/authorized_keys",
    "cp /usr/bin/ssh /tmp/ssh_wrapper", "echo '#!/bin/sh' > /tmp/wrapper.sh", "chmod +x /tmp/wrapper.sh",
    "echo 'alias ls=\"ls -la\"' > ~/.bashrc", "echo 'alias rm=\"rm -i\"' > ~/.bashrc",
    "echo 'alias netstat=\"ss\"' > ~/.bashrc", "echo 'alias ps=\"ps -fe\"' > ~/.bashrc",
    "echo 'alias df=\"df -h\"' > ~/.bashrc", "echo 'alias du=\"du -sh\"' > ~/.bashrc",
    "cp /etc/cron.d/anacron /etc/cron.d/anacron.bak", "chmod 400 /etc/sudoers",
    "chown root:root /etc/sudoers", "touch /var/www/.htaccess",
    "echo 'deny from all' > /var/www/.htaccess", "echo 'order allow,deny' >> /var/www/.htaccess",
    "echo 'Allow from 127.0.0.1' >> /var/www/.htaccess", "echo 'RewriteEngine On' > /var/www/html/.htaccess",
    "echo 'RewriteRule .* - [F]' >> /var/www/html/.htaccess", "echo 'Header set X-Frame-Options DENY' > /etc/apache2/conf-available/security.conf",
    "echo 'ServerSignature Off' >> /etc/apache2/conf-available/security.conf", "echo 'ServerTokens Prod' >> /etc/apache2/conf-available/security.conf",
    "echo 'X-XSS-Protection \"1; mode=block\"' >> /etc/apache2/conf-available/security.conf",
]
def run_kfold_validation(n_splits=5):
    """
    Performs K-Fold Cross-Validation on the action scripts.
    The agent is trained on a subset of actions and tested on a held-out set.
    """
    print(f"\n--- Starting {n_splits}-Fold Cross-Validation on Action Scripts ---")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    
    # Create a dummy log dataset for the environment to reset to, since the focus is on action scripts
    log_dataset = [np.random.randint(50, 150, size=N_FEATURES_LOG) for _ in range(50)]

    for i, (train_index, test_index) in enumerate(kf.split(ACTION_SCRIPTS)):
        print(f"\n--- Fold {i+1}/{n_splits} ---")
        
        # Split actions for this fold
        train_actions = [ACTION_SCRIPTS[j] for j in train_index]
        test_actions = [ACTION_SCRIPTS[j] for j in test_index]
        
        # Train the agent on the training set of actions
        agent = AdvancedNLPAgent(train_actions)
        env = AdvancedLinuxEnvironment(selected_malicious_log, trained_classifier_model, kmeans_model)
        
        EPISODES = 50
        ALPHA = 0.1
        GAMMA = 0.9
        EPSILON = 0.2
        POLICY_TYPE = "epsilon-greedy"
        
        train_rewards = []
        train_malicious_flags = []
        
        for episode in range(EPISODES):
            initial_log = random.choice(log_dataset)
            state = env.reset(initial_log)
            done = False
            total_reward = 0
            malicious_count = 0
            
            while not done:
                action_idx = agent.choose_action(state, EPSILON, policy_type=POLICY_TYPE)
                next_state, reward, done = env.step(train_actions[action_idx])
                agent.update_policy(state, action_idx, reward, next_state, ALPHA, GAMMA)
                state = next_state
                total_reward += reward
                if env.is_malicious:
                    malicious_count += 1
            train_rewards.append(total_reward)
            train_malicious_flags.append(malicious_count)
            
        # Evaluate the trained agent on the test set of actions
        test_rewards = []
        test_malicious_flags = []
        
        for action_script in test_actions:
            initial_log = random.choice(log_dataset)
            state = env.reset(initial_log)
            next_state, reward, done = env.step(action_script)
            test_rewards.append(reward)
            test_malicious_flags.append(1 if env.is_malicious else 0)
            
        avg_test_reward = np.mean(test_rewards)
        avg_test_malicious_rate = np.mean(test_malicious_flags)
        
        print(f"  Fold {i+1} Training Complete. Average Test Reward: {avg_test_reward:.2f}, Malicious Rate: {avg_test_malicious_rate:.2f}")
        
        fold_results.append({
            'train_rewards': train_rewards,
            'train_malicious_flags': train_malicious_flags,
            'test_reward': avg_test_reward,
            'test_malicious_rate': avg_test_malicious_rate,
            'q_table': agent.q_table,
            'actions': train_actions # Store the actions used for this fold
        })
    return fold_results

kfold_results = run_kfold_validation(n_splits=5)
print("\n--- Final Q-Table (from the last fold) ---")
last_fold_results = kfold_results[-1]
final_q_table = last_fold_results['q_table']
final_actions = last_fold_results['actions'] # Retrieve the stored actions
states = sorted(final_q_table.keys())
print(f"{'State':<30} | {' | '.join(f'{action:<20}' for action in final_actions[:5])} ...")
print("-" * (30 + 3 + len(final_actions[:5]) * 23))
for state in states:
    q_values = final_q_table[state]
    q_value_str = ' | '.join(f'{q:.2f}' for q in q_values[:5])
    print(f"{state:<30} | {q_value_str} ...")

print("\n--- Generating Visual Comparisons ---")

# Training Rewards Plot
plt.figure(figsize=(12, 7))
for i, result in enumerate(kfold_results):
    plt.plot(result['train_rewards'], label=f'Fold {i+1} Training Rewards')
plt.title('Total Training Reward per Episode (K-Fold Validation)')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend()
plt.grid(True)
plt.show()

# Training Malicious Flags Plot
plt.figure(figsize=(12, 7))
for i, result in enumerate(kfold_results):
    plt.plot(result['train_malicious_flags'], label=f'Fold {i+1} Malicious Detections')
plt.title('Malicious Detections per Episode (K-Fold Validation)')
plt.xlabel('Episode')
plt.ylabel('Number of Malicious Detections')
plt.legend()
plt.grid(True)
plt.show()

# Average Test Metrics Plot
test_rewards = [res['test_reward'] for res in kfold_results]
test_malicious_rates = [res['test_malicious_rate'] for res in kfold_results]
