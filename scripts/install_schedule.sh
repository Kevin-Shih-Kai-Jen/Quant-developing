#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# install_schedule.sh — 一鍵安裝/移除 Nexus Quant OS 自動排程
# ═══════════════════════════════════════════════════════════════════
#
# Usage:
#   bash scripts/install_schedule.sh install   # 安裝排程
#   bash scripts/install_schedule.sh uninstall # 移除排程
#   bash scripts/install_schedule.sh status    # 檢查狀態
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

PROJECT_DIR="/Users/coolguy/developer/nexus_quant_os"
SCRIPTS_DIR="${PROJECT_DIR}/scripts"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
LOG_DIR="${PROJECT_DIR}/logs"

PREMARKET_LABEL="com.nexusquant.premarket"
POSTMARKET_LABEL="com.nexusquant.postmarket"
PREMARKET_PLIST="${SCRIPTS_DIR}/${PREMARKET_LABEL}.plist"
POSTMARKET_PLIST="${SCRIPTS_DIR}/${POSTMARKET_LABEL}.plist"

ACTION="${1:-status}"

echo "═══════════════════════════════════════════════════════════"
echo "  🔧 Nexus Quant OS — Schedule Manager"
echo "═══════════════════════════════════════════════════════════"
echo ""

case "$ACTION" in
    install)
        echo "  ▸ Installing scheduled jobs..."
        echo ""

        # Ensure directories exist
        mkdir -p "$LAUNCH_AGENTS_DIR"
        mkdir -p "$LOG_DIR"

        # Make run_scheduled.sh executable
        chmod +x "${SCRIPTS_DIR}/run_scheduled.sh"

        # Copy plist files to LaunchAgents
        cp "$PREMARKET_PLIST"  "${LAUNCH_AGENTS_DIR}/${PREMARKET_LABEL}.plist"
        cp "$POSTMARKET_PLIST" "${LAUNCH_AGENTS_DIR}/${POSTMARKET_LABEL}.plist"

        # Load the jobs
        launchctl load "${LAUNCH_AGENTS_DIR}/${PREMARKET_LABEL}.plist" 2>/dev/null || true
        launchctl load "${LAUNCH_AGENTS_DIR}/${POSTMARKET_LABEL}.plist" 2>/dev/null || true

        echo "  ✅ Pre-market job installed  → 每日 AEST 00:25 (美股開盤前)"
        echo "  ✅ Post-market job installed → 每日 AEST 07:10 (美股收盤後)"
        echo ""

        # ── Schedule Mac to wake from sleep ───────────────────────
        echo "  ▸ Setting Mac to wake from sleep for pre-market..."
        echo "    (需要系統密碼)"
        echo ""

        # macOS pmset repeat 只允許一組 wakeorpoweron，
        # 所以只設 00:20（開盤前下單最關鍵）。
        # 收盤後 07:10 的快照：若 Mac 在睡眠，launchd 會在
        # 下次喚醒時自動補跑（StartCalendarInterval 特性）。
        sudo pmset repeat wakeorpoweron MTWRFSU 00:20:00 || {
            echo "    ⚠️  pmset failed — Mac 可能不會在睡眠中自動喚醒"
            echo "       手動設定: System Settings → Energy → Schedule"
        }

        echo ""
        echo "  ✅ 安裝完成！"
        echo ""
        echo "  📋 排程:"
        echo "    • 開盤前 AEST 00:25 → Pipeline + 自動下單 + Discord 報告"
        echo "    • 收盤後 AEST 07:10 → 持倉快照 + Discord 回報"
        echo ""
        echo "  💡 提示:"
        echo "    • 螢幕關起來 (sleep) 也會自動喚醒執行"
        echo "    • 需要 OpenD 在背景運行才能下單"
        echo "    • 日誌: ${LOG_DIR}/"
        echo "    • 移除: bash scripts/install_schedule.sh uninstall"
        ;;

    uninstall)
        echo "  ▸ Removing scheduled jobs..."
        echo ""

        launchctl unload "${LAUNCH_AGENTS_DIR}/${PREMARKET_LABEL}.plist" 2>/dev/null || true
        launchctl unload "${LAUNCH_AGENTS_DIR}/${POSTMARKET_LABEL}.plist" 2>/dev/null || true

        rm -f "${LAUNCH_AGENTS_DIR}/${PREMARKET_LABEL}.plist"
        rm -f "${LAUNCH_AGENTS_DIR}/${POSTMARKET_LABEL}.plist"

        echo "  ✅ Scheduled jobs removed"
        echo ""
        echo "  💡 如果也要移除自動喚醒，執行:"
        echo "     sudo pmset repeat cancel"
        ;;

    status)
        echo "  ▸ Checking schedule status..."
        echo ""

        echo "  Pre-market (${PREMARKET_LABEL}):"
        if launchctl list | grep -q "$PREMARKET_LABEL"; then
            echo "    ✅ LOADED — 每日 AEST 00:25"
        else
            echo "    ❌ NOT LOADED"
        fi
        echo ""

        echo "  Post-market (${POSTMARKET_LABEL}):"
        if launchctl list | grep -q "$POSTMARKET_LABEL"; then
            echo "    ✅ LOADED — 每日 AEST 07:10"
        else
            echo "    ❌ NOT LOADED"
        fi
        echo ""

        echo "  Mac wake schedule:"
        pmset -g sched 2>/dev/null || echo "    (no schedule set)"
        echo ""

        # Show recent logs
        echo "  Recent logs:"
        ls -lt "$LOG_DIR"/scheduled_*.log 2>/dev/null | head -3 || echo "    (none)"
        ;;

    *)
        echo "  Usage: $0 {install|uninstall|status}"
        exit 1
        ;;
esac

echo ""
echo "═══════════════════════════════════════════════════════════"
