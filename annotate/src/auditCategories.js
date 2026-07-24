const EXPLICIT_CATEGORIES = [
  {
    id: 'physical',
    flag: 'physical_must_alert',
    reasons: 'physical_reasons',
    label: 'Physical must-alert',
    reasonPrefix: 'physical must-alert',
    priority: 110,
  },
  {
    id: 'integrity',
    flag: 'data_integrity_must_resolve',
    reasons: 'data_integrity_reasons',
    label: 'Data integrity',
    reasonPrefix: 'data integrity',
    priority: 105,
  },
  {
    id: 'noinfo_context',
    flag: 'noinfo_context_alert',
    reasons: 'noinfo_context_reasons',
    label: 'Near Noinfo region',
    reasonPrefix: 'near Noinfo region',
    priority: 98,
  },
];


function reasonList(value) {
  if (Array.isArray(value)) return value.filter(Boolean);
  if (typeof value === 'string' && value.trim()) {
    return value.split(',').map(reason => reason.trim()).filter(Boolean);
  }
  return [];
}


export function hardReasonsForNote(note) {
  return reasonList(note?.hard_reasons);
}


export function auditCategoryForNote(note) {
  if (!note) return null;
  const category = EXPLICIT_CATEGORIES.find(item => !!note[item.flag]);
  if (!category) return null;
  const reasons = reasonList(note[category.reasons]);
  return {
    id: category.id,
    label: category.label,
    reasons,
    reason: reasons.length > 0
      ? `${category.reasonPrefix}: ${reasons.join(', ')}`
      : category.reasonPrefix,
    priority: category.priority,
  };
}


export function isExplicitAuditNote(note) {
  return auditCategoryForNote(note) !== null;
}


export function isAuditNote(note) {
  return isExplicitAuditNote(note)
    || !!note?.is_hard
    || hardReasonsForNote(note).length > 0;
}


export function explicitAuditPriority(note) {
  const category = auditCategoryForNote(note);
  if (!category) return null;
  return {
    score: category.priority,
    reason: category.reason,
  };
}
