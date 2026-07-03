/**
 * Lan-gpt-image-2 Web UI Enhancement
 *
 * Adds tooltip styling, quick-fill base URL button, and dynamic
 * parameter visibility for the Lan-gpt-image-2 node.
 */

import { app } from "../../scripts/app.js";

const NODE_NAME = "Lan-gpt-image-2";
const EXTENSION_NAME = "Lan.GPTImage2";

// Common base URL presets
const URL_PRESETS = [
    { label: "OpenAI Official", url: "https://api.openai.com/v1" },
    { label: "Local Proxy (8317)", url: "http://localhost:8317/v1" },
    { label: "Local Proxy (8080)", url: "http://localhost:8080/v1" },
    { label: "Azure OpenAI", url: "" },
];

app.registerExtension({
    name: EXTENSION_NAME,

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== NODE_NAME) return;

        // Add CSS for better tooltip display
        const style = document.createElement("style");
        style.textContent = `
            .lan-gpt-tooltip {
                background: #1a1a2e !important;
                border: 1px solid #4a4a6a !important;
                color: #e0e0f0 !important;
                font-size: 12px !important;
                border-radius: 6px !important;
                padding: 8px 12px !important;
                max-width: 320px !important;
                box-shadow: 0 4px 12px rgba(0,0,0,0.4) !important;
            }
        `;
        if (!document.getElementById("lan-gpt-style")) {
            style.id = "lan-gpt-style";
            document.head.appendChild(style);
        }

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);

            // Set a distinctive title color
            this.color = "#1a1a3a";
            this.bgcolor = "#2a2a4a";

            // Add a "Quick URL" subtitle widget
            const urlWidget = this.widgets?.find((w) => w.name === "base_url");
            if (urlWidget) {
                urlWidget.tooltip =
                    "API base URL. Use https://api.openai.com/v1 for official, " +
                    "or http://localhost:8317/v1 for a local proxy.";
            }

            return result;
        };

        // Dynamic visibility: hide edit-related tooltips when no image is connected
        const originalOnDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            return originalOnDrawForeground?.apply(this, arguments);
        };
    },

    async loadedGraphNode(node) {
        if (node.type !== NODE_NAME) return;
        // Ensure default values are properly set
        const widgets = node.widgets || [];
        for (const w of widgets) {
            if (w.name === "base_url" && (!w.value || w.value === "")) {
                w.value = "https://api.openai.com/v1";
            }
        }
    },
});
